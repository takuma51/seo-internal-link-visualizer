import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network

# ========= 設定 =========
INPUT = Path("data/all_outlinks.csv")
OUTPUT = Path("docs/index.html")

TARGET_DOMAIN = "okatakuma.tokyo"

ONLY_STATUS_200 = True
ONLY_FOLLOW_TRUE = True
DROP_SAME_URL_EDGE = True

DROP_QUERY_AND_FRAGMENT = True
DROP_TRAILING_SLASH = True

TOP_N_NODES = 250          # ← まずは 150〜250 推奨（400は重くなりがち）
MAX_EDGES = 4000           # ← エッジ多すぎると100%止まるので上限を入れる

# PyVis（vis-network）
HEIGHT = "900px"
WIDTH = "100%"
BG_COLOR = "#0b0f19"
FONT_COLOR = "#ffffff"
SHOW_LABEL = False         # URLラベルは重いので基本OFF
# =======================


def normalize_url(url: str) -> str:
    if url is None:
        return ""
    url = str(url).strip()
    if not url:
        return ""
    url = re.sub(r"\s+", "", url)

    try:
        p = urlparse(url)
    except Exception:
        return url

    scheme = (p.scheme or "https").lower()
    netloc = (p.netloc or "").lower()
    path = p.path or "/"

    query = "" if DROP_QUERY_AND_FRAGMENT else (p.query or "")
    fragment = "" if DROP_QUERY_AND_FRAGMENT else (p.fragment or "")

    if DROP_TRAILING_SLASH and path != "/":
        path = path.rstrip("/")

    return urlunparse((scheme, netloc, path, "", query, fragment))


def host_of(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def is_internal(url: str) -> bool:
    h = host_of(url)
    return (h == TARGET_DOMAIN) or h.endswith("." + TARGET_DOMAIN)


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def pagerank_power_iteration(G: nx.DiGraph, alpha: float = 0.85, max_iter: int = 200, tol: float = 1e-6) -> dict:
    """
    SciPy不要のPageRank（numpyだけでpower iteration）
    """
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    out_deg = np.zeros(n, dtype=float)
    for u in nodes:
        out_deg[idx[u]] = G.out_degree(u)

    # 初期値
    pr = np.ones(n, dtype=float) / n
    teleport = (1.0 - alpha) / n

    # 隣接（in-links）で更新
    in_neighbors = [[] for _ in range(n)]
    for u, v in G.edges():
        in_neighbors[idx[v]].append(idx[u])

    for _ in range(max_iter):
        prev = pr.copy()

        # ダングリング（外向き0）ノードのPRは全体に分配
        dangling_sum = prev[out_deg == 0].sum()

        for i in range(n):
            s = 0.0
            for j in in_neighbors[i]:
                if out_deg[j] > 0:
                    s += prev[j] / out_deg[j]
            pr[i] = teleport + alpha * (s + dangling_sum / n)

        # 収束判定（L1ノルム）
        if np.abs(pr - prev).sum() < tol:
            break

    return {nodes[i]: float(pr[i]) for i in range(n)}


def main():
    df = pd.read_csv(INPUT, low_memory=False)

    col_source = pick_col(df, ["Source", "source", "From", "URL Source", "Source URL"])
    col_dest = pick_col(df, ["Destination", "destination", "To", "URL Destination", "Destination URL"])
    if not col_source or not col_dest:
        raise ValueError(f"Source/Destination相当の列が見つかりません。列一覧: {list(df.columns)}")

    # URL正規化
    df[col_source] = df[col_source].astype(str).map(normalize_url)
    df[col_dest] = df[col_dest].astype(str).map(normalize_url)
    df = df[(df[col_source] != "") & (df[col_dest] != "")]

    # 内部リンクのみに絞る
    df = df[df[col_source].map(is_internal)]
    df = df[df[col_dest].map(is_internal)]

    # ステータス200
    col_status = pick_col(df, ["Status Code", "Status", "HTTP Status Code", "Response Code"])
    if ONLY_STATUS_200 and col_status:
        df = df[pd.to_numeric(df[col_status], errors="coerce") == 200]

    # follow
    col_follow = pick_col(df, ["Follow", "follow", "Is Follow", "Link Follow"])
    if ONLY_FOLLOW_TRUE and col_follow:
        df = df[to_bool_series(df[col_follow])]

    # 自己ループ除外
    if DROP_SAME_URL_EDGE:
        df = df[df[col_source] != df[col_dest]]

    edges = df[[col_source, col_dest]].dropna().rename(columns={col_source: "Source", col_dest: "Destination"})
    if edges.empty:
        raise ValueError("エッジが0件です。フィルタが厳しすぎる可能性があります。")

    # グラフ構築
    G = nx.from_pandas_edgelist(edges, source="Source", target="Destination", create_using=nx.DiGraph())
    if G.number_of_nodes() == 0:
        raise ValueError("ノードが0です。CSVやフィルタ条件を見直してください。")

    # PageRank（scipy不要）
    pr = pagerank_power_iteration(G, alpha=0.85, max_iter=200, tol=1e-6)

    # 上位ノードだけ残す
    if TOP_N_NODES is not None and G.number_of_nodes() > TOP_N_NODES:
        top_nodes = [n for n, _ in sorted(pr.items(), key=lambda x: x[1], reverse=True)[:TOP_N_NODES]]
        top_set = set(top_nodes)
        G = G.subgraph(top_set).copy()
        pr = {n: pr[n] for n in G.nodes()}

    # エッジ上限（重さ対策）
    if MAX_EDGES is not None and G.number_of_edges() > MAX_EDGES:
        # “重要ノード同士のエッジ”を優先して残す（単純スコア）
        scored = []
        for s, t in G.edges():
            scored.append((pr.get(s, 0.0) + pr.get(t, 0.0), s, t))
        scored.sort(reverse=True)
        keep = scored[:MAX_EDGES]
        G2 = nx.DiGraph()
        G2.add_nodes_from(G.nodes())
        G2.add_edges_from([(s, t) for _, s, t in keep])
        G = G2

    # PyVis出力
    net = Network(height=HEIGHT, width=WIDTH, directed=True, bgcolor=BG_COLOR, font_color=FONT_COLOR)
    net.barnes_hut(gravity=-8000, central_gravity=0.2, spring_length=180, spring_strength=0.02, damping=0.25)

    # stabilization（ここが「0%」の正体）を制限して終わらせる
    net.set_options(
        """
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": { "enabled": true, "iterations": 200, "updateInterval": 25 }
          },
          "nodes": {
            "shape": "dot"
          },
          "edges": {
            "smooth": false
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """
    )

    max_pr = max(pr.values()) if pr else 1.0
    for n in G.nodes():
        score = (pr.get(n, 0.0) / max_pr) if max_pr else 0.0
        size = 8 + (score * 35)  # 8〜43くらい
        label = n if SHOW_LABEL else ""
        net.add_node(n, label=label, title=f"{n}<br>PageRank: {pr.get(n, 0.0):.8f}", size=size)

    for s, t in G.edges():
        net.add_edge(s, t)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(OUTPUT))

    print(f"Saved: {OUTPUT}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()
