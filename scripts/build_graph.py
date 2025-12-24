import pandas as pd
import networkx as nx
from pyvis.network import Network
from pathlib import Path
from urllib.parse import urlparse

# ========= 設定 =========
INPUT = Path("data/all_outlinks.csv")
OUTPUT = Path("docs/index.html")

# 対象ドメイン（内部リンク判定用）
TARGET_DOMAIN = "okatakuma.tokyo"

# フィルタ/調整
ONLY_STATUS_200 = True          # True: 200のみ / False: 全て
ONLY_FOLLOW_TRUE = True         # True: Follow列がtrueのみ（SF出力にある場合）
DROP_SAME_URL_EDGE = True       # source==destination を除外
TOP_N_NODES = 400               # 重い場合は下げる（例: 200）。Noneで無制限

# 可視化（PyVis）
HEIGHT = "800px"
WIDTH = "100%"
BG_COLOR = "#0b0f19"
FONT_COLOR = "#ffffff"
# =======================


def is_internal(url: str, target_domain: str) -> bool:
    """同一ドメイン（サブドメイン含む）を内部リンクとして判定します。"""
    try:
        host = urlparse(url).netloc.lower()
        return host == target_domain or host.endswith("." + target_domain)
    except Exception:
        return False


def main():
    # 1) 読み込み
    df = pd.read_csv(INPUT)

    # 2) 必須列チェック（Screaming Frog Outlinks想定）
    required_cols = {"Source", "Destination"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSVに必要な列がありません: {missing}. いまある列: {list(df.columns)}")

    # 3) 内部リンクに絞る
    df = df[df["Source"].astype(str).apply(lambda x: is_internal(x, TARGET_DOMAIN))]
    df = df[df["Destination"].astype(str).apply(lambda x: is_internal(x, TARGET_DOMAIN))]

    # 4) 追加フィルタ（列がある時だけ適用）
    if ONLY_STATUS_200 and "Status Code" in df.columns:
        df = df[df["Status Code"] == 200]

    if ONLY_FOLLOW_TRUE and "Follow" in df.columns:
        # Screaming Frogは true/false 文字列のことが多いので吸収します
        df = df[df["Follow"].astype(str).str.lower().isin(["true", "1", "yes"])]

    if DROP_SAME_URL_EDGE:
        df = df[df["Source"] != df["Destination"]]

    # 5) グラフ構築（有向）
    G = nx.from_pandas_edgelist(
        df,
        source="Source",
        target="Destination",
        create_using=nx.DiGraph()
    )

    if G.number_of_nodes() == 0:
        raise ValueError("ノードが0です。フィルタが厳しすぎる可能性があります。")

    # 6) PageRank（重要度）
    pr = nx.pagerank(G, alpha=0.85)
    nx.set_node_attributes(G, pr, "pagerank")

    # 7) 重い場合のノード削減（上位Nのみ表示）
    if TOP_N_NODES is not None and G.number_of_nodes() > TOP_N_NODES:
        top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:TOP_N_NODES]
        top_nodes = set([n for n, _ in top_nodes])
        G = G.subgraph(top_nodes).copy()
        pr = {n: pr[n] for n in G.nodes()}

    # 8) PyVisでHTML出力
    net = Network(height=HEIGHT, width=WIDTH, directed=True, bgcolor=BG_COLOR, font_color=FONT_COLOR)
    net.barnes_hut()

    # ノード追加（PageRankに応じてサイズ調整）
    max_pr = max(pr.values()) if pr else 1.0
    for n in G.nodes():
        # 0〜1へ正規化 → 見やすいサイズへ
        score = pr.get(n, 0) / max_pr if max_pr else 0
        size = 8 + (score * 40)  # 8〜48くらい
        net.add_node(n, label="", title=f"{n}<br>PageRank: {pr.get(n, 0):.6f}", size=size)

    # エッジ追加
    for s, t in G.edges():
        net.add_edge(s, t)

    # 保存
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(OUTPUT))

    print(f"Saved: {OUTPUT}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")


if __name__ == "__main__":
    main()
