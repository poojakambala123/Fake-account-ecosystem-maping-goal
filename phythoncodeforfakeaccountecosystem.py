import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import community as community_louvain  # python-louvain
from pyvis.network import Network
import datetime
import random
from itertools import combinations

# -------------------------
# Synthetic dataset helper
# -------------------------
def generate_synthetic_accounts(n=200, seed=42):
    """
    Generate synthetic account records.
    Fields:
      account_id, created_at (datetime), ip, email, friends (list of account_ids), posts_per_day
    We'll simulate pockets of coordinated fake accounts sharing IP/email domain/friend groups.
    """
    random.seed(seed)
    np.random.seed(seed)
    base_date = datetime.datetime(2024, 1, 1)

    accounts = []
    # Create several malicious clusters
    clusters = []
    for c in range(5):
        size = random.randint(8, 20)
        cluster_ids = [f"acct_c{c}_{i}" for i in range(size)]
        clusters.append(cluster_ids)

    # remainder are benign
    rest = [f"acct_b_{i}" for i in range(n - sum(len(x) for x in clusters))]

    all_ids = [i for c in clusters for i in c] + rest

    # create cluster-level shared attributes
    cluster_ips = [f"10.0.{c}.1" for c in range(len(clusters))]
    cluster_domains = [f"spamdom{c}.xyz" for c in range(len(clusters))]

    # create accounts
    for idx, aid in enumerate(all_ids):
        if aid.startswith("acct_c"):
            # find cluster index
            c = int(aid.split('_')[1][1:])
            created_at = base_date + datetime.timedelta(days=random.randint(0,10))  # bursty creation
            ip = cluster_ips[c]
            email = f"user{random.randint(1,100)}@{cluster_domains[c]}"
            posts_per_day = max(0, np.random.normal(5, 2))  # more regular posting
            # friends: connect heavily within cluster + few outside
            friends = random.sample([x for x in clusters[c] if x!=aid], k=min(5, len(clusters[c])-1))
            # add occasional cross cluster friend
            if random.random() < 0.2:
                other_cluster = random.choice([cc for cc in [0,1,2,3,4] if cc!=c])
                friends.append(random.choice(clusters[other_cluster]))
        else:
            created_at = base_date + datetime.timedelta(days=random.randint(0,365))
            ip = f"192.168.{random.randint(0,100)}.{random.randint(2,254)}"
            dom = random.choice(["gmail.com","hotmail.com","example.org","yahoo.com"])
            email = f"human{random.randint(1,10000)}@{dom}"
            posts_per_day = max(0, np.random.normal(0.4, 0.8))
            # benign friend graph is sparse & random
            friends = random.sample(all_ids, k=min(3, max(0, int(np.random.poisson(1)))))
            friends = [f for f in friends if f != aid]
        accounts.append({
            "account_id": aid,
            "created_at": created_at,
            "ip": ip,
            "email": email,
            "friends": list(set(friends)),
            "posts_per_day": posts_per_day
        })
    return pd.DataFrame(accounts)


# -------------------------
# Feature engineering
# -------------------------
def account_similarity_features(df):
    """
    Build pairwise features between accounts.
    Returns DataFrame with columns: a, b, shared_ip, same_email_domain, friend_overlap, creation_time_diff_days, posts_diff
    """
    # Precompute helper maps
    id_to_friends = {row.account_id: set(row.friends) for row in df.itertuples()}
    id_to_ip = {row.account_id: row.ip for row in df.itertuples()}
    id_to_email = {row.account_id: row.email for row in df.itertuples()}
    id_to_created = {row.account_id: row.created_at for row in df.itertuples()}
    id_to_posts = {row.account_id: row.posts_per_day for row in df.itertuples()}

    records = []
    ids = df['account_id'].tolist()
    # For performance, only consider pairs with at least some relationship (friend link or same ip/email) + sample otherwise
    # But for simplicity we compute all pairs here (O(n^2)). For large n, sample or use blocking.
    for a, b in combinations(ids, 2):
        shared_ip = 1 if id_to_ip[a] == id_to_ip[b] else 0
        # email domain
        dom_a = id_to_email[a].split('@')[-1]
        dom_b = id_to_email[b].split('@')[-1]
        same_email_domain = 1 if dom_a == dom_b else 0
        # friend overlap Jaccard
        fa = id_to_friends.get(a, set())
        fb = id_to_friends.get(b, set())
        inter = len(fa & fb)
        union = len(fa | fb)
        friend_overlap = inter / union if union > 0 else 0.0
        # creation time difference in days
        ct_a = id_to_created[a]
        ct_b = id_to_created[b]
        creation_time_diff_days = abs((ct_a - ct_b).days)
        posts_diff = abs(id_to_posts[a] - id_to_posts[b])
        # also a direct friend link indicator
        direct_friend = 1 if (b in fa or a in fb) else 0

        records.append({
            "a": a, "b": b,
            "shared_ip": shared_ip,
            "same_email_domain": same_email_domain,
            "friend_overlap": friend_overlap,
            "creation_time_diff_days": creation_time_diff_days,
            "posts_diff": posts_diff,
            "direct_friend": direct_friend
        })
    return pd.DataFrame(records)


# -------------------------
# Build graph
# -------------------------
def build_weighted_graph(df_pairs, weight_config=None, min_weight_threshold=0.1):
    """
    Build NetworkX graph from pairwise feature DataFrame.
    weight_config: dict mapping feature -> weight for combining into edge weight
    """
    if weight_config is None:
        weight_config = {
            "shared_ip": 3.0,
            "same_email_domain": 2.0,
            "friend_overlap": 5.0,
            "direct_friend": 1.0,
            # lower weight for similarity by creation time and posts diff (we transform)
            "creation_time_sim": 1.5,
            "posts_sim": 1.0
        }

    # normalize creation_time_diff and posts_diff to similarity [0,1]
    max_days = max(1, df_pairs['creation_time_diff_days'].max())
    df_pairs['creation_time_sim'] = 1 - (df_pairs['creation_time_diff_days'] / (max_days + 1))
    max_posts_diff = max(1e-6, df_pairs['posts_diff'].max())
    df_pairs['posts_sim'] = 1 - (df_pairs['posts_diff'] / (max_posts_diff + 1))

    # compute weight
    def combined_weight(row):
        s = 0.0
        s += weight_config.get("shared_ip",0) * row['shared_ip']
        s += weight_config.get("same_email_domain",0) * row['same_email_domain']
        s += weight_config.get("friend_overlap",0) * row['friend_overlap']
        s += weight_config.get("direct_friend",0) * row['direct_friend']
        s += weight_config.get("creation_time_sim",0) * row['creation_time_sim']
        s += weight_config.get("posts_sim",0) * row['posts_sim']
        return s

    df_pairs['weight'] = df_pairs.apply(combined_weight, axis=1)

    G = nx.Graph()
    # Add nodes (we'll add attributes later externally)
    nodes = set(df_pairs['a']).union(set(df_pairs['b']))
    G.add_nodes_from(nodes)
    # Add edges above threshold
    for r in df_pairs.itertuples():
        if r.weight >= min_weight_threshold:
            G.add_edge(r.a, r.b, weight=r.weight,
                       shared_ip=r.shared_ip,
                       same_email_domain=r.same_email_domain,
                       friend_overlap=r.friend_overlap,
                       direct_friend=r.direct_friend,
                       creation_time_diff_days=r.creation_time_diff_days,
                       posts_diff=r.posts_diff)
    return G, df_pairs


# -------------------------
# Community detection + scoring
# -------------------------
def detect_communities(G):
    """
    Use Louvain community detection. Returns dict node->community and compact summary.
    """
    # use weight attribute for Louvain
    part = community_louvain.best_partition(G, weight='weight')
    # build community summary
    comm_df = pd.DataFrame(list(part.items()), columns=['node', 'community'])
    sizes = comm_df.groupby('community').size().reset_index(name='size').sort_values('size', ascending=False)
    return part, sizes, comm_df


def score_accounts(df_accounts, G, pairs_df, isolation_n_estimators=100):
    """
    Create feature vectors per account (degree, avg edge weight, clustering, fraction_shared_ip_friends, posts_per_day)
    Then compute two suspiciousness signals:
      - heuristic_score: weighted linear combination of suspicious features
      - iso_score: IsolationForest anomaly score (higher -> more anomalous)
    """
    nodes = list(G.nodes())
    deg = dict(G.degree(weight=None))
    weighted_deg = dict(G.degree(weight='weight'))
    clustering = nx.clustering(G, weight='weight')

    rows = []
    # compute fraction of neighbor edges that have shared_ip or same_email_domain
    for n in nodes:
        nbrs = G[n]
        edges = nbrs.items()
        if len(edges) == 0:
            avg_weight = 0.0
            frac_shared_ip = 0.0
            frac_same_email_dom = 0.0
        else:
            weights = [d.get('weight',0) for _, d in edges]
            avg_weight = np.mean(weights)
            shared_ips = [d.get('shared_ip',0) for _, d in edges]
            same_dom = [d.get('same_email_domain',0) for _, d in edges]
            frac_shared_ip = sum(shared_ips) / len(shared_ips)
            frac_same_email_dom = sum(same_dom) / len(same_dom)
        rows.append({
            'account_id': n,
            'degree': deg.get(n,0),
            'weighted_degree': weighted_deg.get(n,0.0),
            'avg_edge_weight': avg_weight,
            'clustering': clustering.get(n,0.0),
            'frac_shared_ip': frac_shared_ip,
            'frac_same_email_dom': frac_same_email_dom
        })
    feat_df = pd.DataFrame(rows)
    # merge posts_per_day and created_at from original
    base_df = df_accounts.set_index('account_id')
    feat_df = feat_df.set_index('account_id').join(base_df[['posts_per_day','created_at','ip','email']])
    feat_df.reset_index(inplace=True)

    # heuristic score (example weights — tune for your data)
    feat_df['heuristic_score'] = (
        2.5 * feat_df['frac_shared_ip'] +
        1.5 * feat_df['frac_same_email_dom'] +
        0.8 * (feat_df['avg_edge_weight'] / (1 + feat_df['avg_edge_weight'])) +
        0.5 * (1 - feat_df['clustering'])  # low clustering (star-like) may be suspicious
    )

    # Isolation Forest for anomaly detection
    iso_feats = feat_df[['degree','weighted_degree','avg_edge_weight','clustering','frac_shared_ip','frac_same_email_dom','posts_per_day']].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(iso_feats)
    iso = IsolationForest(n_estimators=isolation_n_estimators, contamination=0.03, random_state=42)
    iso.fit(Xs)
    # decision_function: higher -> less anomalous; we invert to produce score where higher == more suspicious
    anomaly_score = -iso.decision_function(Xs)
    feat_df['iso_score'] = anomaly_score
    # combine
    feat_df['final_score'] = 0.6 * feat_df['heuristic_score'] + 0.4 * (feat_df['iso_score'] / (np.max(feat_df['iso_score']) + 1e-9))
    feat_df.sort_values('final_score', ascending=False, inplace=True)
    return feat_df


# -------------------------
# Visualization
# -------------------------
def visualize_graph_pyvis(G, feat_df, out_html="ecosystem_map.html", max_nodes=200):
    """
    Save interactive HTML with pyvis. Node size colored by final_score if available.
    """
    net = Network(height="800px", width="100%", notebook=False, bgcolor="#222222", font_color="white")
    # limit nodes if giant graph
    nodes_to_show = list(G.nodes())[:max_nodes]
    # create score map
    score_map = feat_df.set_index('account_id')['final_score'].to_dict() if 'final_score' in feat_df.columns else {}
    max_score = max(score_map.values()) if score_map else 1.0

    for n in nodes_to_show:
        score = score_map.get(n, 0.0)
        size = 8 + (20 * float(score) / (max_score + 1e-9))
        title = f"{n}<br>score={score:.3f}"
        net.add_node(n, label=n, title=title, value=size)

    for u, v, d in G.edges(data=True):
        if u in nodes_to_show and v in nodes_to_show:
            w = d.get('weight', 0.1)
            net.add_edge(u, v, value=w, title=f"w={w:.2f}")

    net.show_buttons(filter_=['physics'])
    net.show(out_html)
    print(f"[+] Wrote interactive visualization to {out_html}")


def visualize_graph_matplotlib(G, feat_df, top_n=100):
    """
    Quick matplotlib static visualization of top_n suspicious nodes subgraph.
    """
    # pick subgraph of top_n by final_score (if available)
    if 'final_score' in feat_df.columns:
        top_nodes = feat_df.head(top_n)['account_id'].tolist()
        subG = G.subgraph(top_nodes).copy()
    else:
        subG = G.copy()
    pos = nx.spring_layout(subG, seed=42, weight='weight')
    scores = {r['account_id']: r.get('final_score',0) for r in feat_df.to_dict('records')}
    node_colors = [scores.get(n,0.0) for n in subG.nodes()]
    nx.draw_networkx_edges(subG, pos, alpha=0.3)
    nodes = nx.draw_networkx_nodes(subG, pos, node_size=80, cmap=plt.cm.viridis,
                                   node_color=node_colors)
    nx.draw_networkx_labels(subG, pos, font_size=6)
    plt.colorbar(nodes, label='suspicious score')
    plt.title("Top suspicious accounts subgraph")
    plt.axis('off')
    plt.show()

# -------------------------
# Example pipeline
# -------------------------
def run_example():
    print("[*] Generating synthetic accounts...")
    df_accounts = generate_synthetic_accounts(n=200)
    print("[*] Computing pairwise similarity features (O(n^2) -- scales badly if n large)")
    pairs = account_similarity_features(df_accounts)
    print("[*] Building weighted graph")
    G, pairs = build_weighted_graph(pairs, min_weight_threshold=0.2)
    print(f"[*] Graph nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
    print("[*] Detecting communities")
    part, sizes, comm_df = detect_communities(G)
    print("[*] Community sizes (top):")
    print(sizes.head(10))
    print("[*] Scoring accounts")
    feat_df = score_accounts(df_accounts, G, pairs)
    print("[*] Top suspicious accounts:")
    print(feat_df[['account_id','final_score','heuristic_score','iso_score']].head(20))
    # attach community label
    feat_df['community'] = feat_df['account_id'].map(part).fillna(-1).astype(int)
    # export
    feat_df.to_csv("account_scores.csv", index=False)
    print("[+] Saved account_scores.csv")
    # visualize
    visualize_graph_pyvis(G, feat_df, out_html="ecosystem_map.html", max_nodes=250)
    visualize_graph_matplotlib(G, feat_df, top_n=60)


if __name__ == "__main__":
    run_example()
