import re
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network

# ========= 設定 =========
INPUT = Path("data/all_outlinks.csv")

# GitHub Pagesに載せる出力
OUTPUT_HTML = Path("docs/index.html")
REPORT_CSV = Path("docs/internal_link_rank.csv")

TARGET_DOMAIN = "okatakuma.tokyo"

ONLY_STATUS_200 = True
ONLY_FOLLOW_TRUE = True
DROP_SAME_URL_EDGE = True

DROP_QUERY_AND_FRAGMENT = True
DROP_TRAILING_SLASH = True

# 重さ対策（可視化だけに効かせる）
TOP_N_NODES = 250
MAX_EDGES = 3000

# 「ページだけ」に寄せる（超重要）
EXCLUDE_PATH_PREFIXES = [
    "/wp-content/uploads/",
    "/wp-includes/",
    "/wp-content/themes/",
    "/wp-content/plugins/",
]
EXCLUDE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".ico",
    ".css", ".js", ".map",
    ".pdf", ".zip", ".rar", ".7z",
    ".mp4", ".mov", ".avi", ".mp3", ".wav",
    ".woff", ".woff2", ".ttf", ".eot",
}

# PyVis
HEIGHT = "900px"
WIDTH = "100%"
BG_COLOR = "#0b0f19"
FONT_COLOR = "#ffffff"
SHOW_LABEL = True  # ページだけに絞ればONでも耐えやすい

# レポートの行数（大きすぎるとCSVが重くなるので上位だけにしたい場合）
REPORT_TOP_N = 300  # None にすると全行出す
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


def path_of(url: str) -> str:
    try:
        return urlparse(url).path or "/"
    except Exception:
        return "/"


def is_internal(url: str) -> bool:
    h = host_of(url)
    return (h == TARGET_DOMAIN) or h.endswith("." + TARGET_DOMAIN)


def is_asset_url(url: str) -> bool:
    p = path_of(url).lower()

    for pref in EXCLUDE_PATH_PREFIXES:
        if p.startswith(pref):
            return True

    m = re.search(r"(\.[a-z0-9]+)$", p)
    if m and m.group(1) in EXCLUDE_EXTENSIONS:
        return True

    return False


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_bool_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


def pagerank_power_iteration(
    G: nx.DiGraph,
    alpha: float = 0.85,
    max_iter: int = 200,
    tol: float = 1e-6
) -> dict[str, float]:
    """SciPy不要のPageRank（numpyだけでpower iteration）"""
    nodes = list(G.nodes())
    n = len(nodes)
    idx = {v: i for i, v in enumerate(nodes)}

    out_deg = np.zeros(n, dtype=float)
    for u in nodes:
        out_deg[idx[u]] = G.out_degree(u)

    pr = np.ones(n, dtype=float) / n
    teleport = (1.0 - alpha) / n

    in_neighbors: list[list[int]] = [[] for _ in range(n)]
    for u, v in G.edges():
        in_neighbors[idx[v]].append(idx[u])

    for _ in range(max_iter):
        prev = pr.copy()
        dangling_sum = prev[out_deg == 0].sum()

        for i in range(n):
            s = 0.0
            for j in in_neighbors[i]:
                if out_deg[j] > 0:
                    s += prev[j] / out_deg[j]
            pr[i] = teleport + alpha * (s + dangling_sum / n)

        if np.abs(pr - prev).sum() < tol:
            break

    return {nodes[i]: float(pr[i]) for i in range(n)}


def short_label(url: str) -> str:
    """ラベルは短く。意味のあるpathだけ表示。"""
    p = path_of(url)
    if p == "/" or p == "":
        return "/"
    p = p.strip("/")
    if len(p) > 45:
        return p[:20] + "…" + p[-20:]
    return p


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

    # 内部リンク
    df = df[df[col_source].map(is_internal)]
    df = df[df[col_dest].map(is_internal)]

    # ページだけ（アセット除外）
    df = df[~df[col_source].map(is_asset_url)]
    df = df[~df[col_dest].map(is_asset_url)]

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

    # ========= 1) 全体グラフ（レポート用） =========
    G_all = nx.from_pandas_edgelist(edges, source="Source", target="Destination", create_using=nx.DiGraph())
    if G_all.number_of_nodes() == 0:
        raise ValueError("ノードが0です。CSVやフィルタ条件を見直してください。")

    pr_all = pagerank_power_iteration(G_all, alpha=0.85, max_iter=200, tol=1e-6)

    # ========= 2) レポートCSV（ここが本体） =========
    report_rows = []
    for node in G_all.nodes():
        report_rows.append({
            "url": node,
            "path": path_of(node),
            "label": short_label(node),
            "pagerank": pr_all.get(node, 0.0),
            "in_links": int(G_all.in_degree(node)),
            "out_links": int(G_all.out_degree(node)),
        })

    report_df = pd.DataFrame(report_rows).sort_values("pagerank", ascending=False)

    if REPORT_TOP_N is not None and len(report_df) > REPORT_TOP_N:
        report_df = report_df.head(REPORT_TOP_N).copy()

    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_CSV, index=False)

    # ========= 3) 可視化用に軽量化 =========
    G_vis = G_all
    pr_vis = pr_all

    if TOP_N_NODES is not None and G_vis.number_of_nodes() > TOP_N_NODES:
        top_nodes = [n for n, _ in sorted(pr_vis.items(), key=lambda x: x[1], reverse=True)[:TOP_N_NODES]]
        top_set = set(top_nodes)
        G_vis = G_vis.subgraph(top_set).copy()
        pr_vis = {n: pr_vis[n] for n in G_vis.nodes()}

    if MAX_EDGES is not None and G_vis.number_of_edges() > MAX_EDGES:
        scored = []
        for s, t in G_vis.edges():
            scored.append((pr_vis.get(s, 0.0) + pr_vis.get(t, 0.0), s, t))
        scored.sort(reverse=True)
        keep = scored[:MAX_EDGES]
        G2 = nx.DiGraph()
        G2.add_nodes_from(G_vis.nodes())
        G2.add_edges_from([(s, t) for _, s, t in keep])
        G_vis = G2

    # ========= 4) HTML生成 =========
    net = Network(height=HEIGHT, width=WIDTH, directed=True, bgcolor=BG_COLOR, font_color=FONT_COLOR)

    net.barnes_hut(gravity=-5000, central_gravity=0.25, spring_length=160, spring_strength=0.03, damping=0.35)

    net.set_options(
        """
        var options = {
          "physics": {
            "enabled": true,
            "stabilization": { "enabled": true, "iterations": 120, "updateInterval": 25 }
          },
          "nodes": { "shape": "dot" },
          "edges": { "smooth": false },
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
          }
        }
        """
    )

    max_pr = max(pr_vis.values()) if pr_vis else 1.0
    for n in G_vis.nodes():
        score = (pr_vis.get(n, 0.0) / max_pr) if max_pr else 0.0
        size = 8 + (score * 35)
        label = short_label(n) if SHOW_LABEL else ""
        net.add_node(
            n,
            label=label,
            title=f"{n}<br>PageRank: {pr_vis.get(n, 0.0):.8f}<br>in:{G_vis.in_degree(n)} out:{G_vis.out_degree(n)}",
            size=size
        )

    for s, t in G_vis.edges():
        net.add_edge(s, t)

    OUTPUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(OUTPUT_HTML))

    print(f"Saved: {OUTPUT_HTML}")
    print(f"Saved report: {REPORT_CSV}")
    print(f"All nodes/edges: {G_all.number_of_nodes()} / {G_all.number_of_edges()}")
    print(f"Vis nodes/edges: {G_vis.number_of_nodes()} / {G_vis.number_of_edges()}")


if __name__ == "__main__":
    main()
