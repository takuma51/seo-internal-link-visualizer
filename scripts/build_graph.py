import pandas as pd
import networkx as nx
from pyvis.network import Network
from pathlib import Path
from urllib.parse import urlparse, urlunparse
import re

# ========= 設定 =========
INPUT = Path("data/all_outlinks.csv")
OUTPUT = Path("docs/index.html")

# 対象ドメイン（内部リンク判定用）
TARGET_DOMAIN = "okatakuma.tokyo"

# フィルタ/調整
ONLY_STATUS_200 = True          # True: 200のみ / False: 全て
ONLY_FOLLOW_TRUE = True         # True: Follow列がtrueのみ（SF出力にある場合）
DROP_SAME_URL_EDGE = True       # source==destination を除外
DROP_QUERY_AND_FRAGMENT = True  # URLの ? と # を落として正規化（重複ノード削減に効く）
DROP_TRAILING_SLASH = True      # 末尾スラッシュを統一（/ と無しの重複回避）
TOP_N_NODES = 400               # 重い場合は下げる（例: 200）。Noneで無制限

# 可視化（PyVis）
HEIGHT = "800px"
WIDTH = "100%"
BG_COLOR = "#0b0f19"
FONT_COLOR = "#ffffff"
SHOW_LABEL = False              # TrueでURLラベルを表示（重くなる）
# =======================


# ---------- ユーティリティ ----------
def normalize_url(url: str) -> str:
    """URLをできるだけ安定した形に正規化します。"""
    if url is None:
        return ""
    url = str(url).strip()
    if not url:
        return ""

    # 空白などを除去（念のため）
    url = re.sub(r"\s+", "", url)

    try:
        parsed = urlparse(url)
    except Exception:
        return url

    scheme = (parsed.scheme or "https").lower()
    netloc = (parsed.netloc or "").lower()
    path = parsed.path or "/"

    # query/fragmentを落とす（設定次第）
    query = "" if DROP_QUERY_AND_FRAGMENT else (parsed.query or "")
    fragment = "" if DROP_QUERY_AND_FRAGMENT else (parsed.fragment or "")

    # 末尾スラッシュ統一（設定次第）
    if DROP_TRAILING_SLASH and path != "/":
        path = path.rstrip("/")

    return urlunparse((scheme, netloc, path, "", query, fragment))


def get_host(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def is_internal(url: str, target_domain: str) -> bool:
    """同一ドメイン（サブドメイン含む）を内部リンクとして判定します。"""
    host = get_host(url)
    return host == target_domain or host.endswith("." + target_domain)


def pick_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """列名ゆらぎ吸収。候補のうち存在する最初の列を返す。"""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def to_bool_str(s: pd.Series) -> pd.Series:
    """true/false/1/0/yes/no のゆらぎを吸収して True/False 判定用のboolへ寄せる。"""
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y"])


# ---------- メイン ----------
def main():
    # 1) 読み込み（巨大CSV対策で low_memory=False）
    df = pd.read_csv(INPUT, low_memory=False)

    # 2) 列名ゆらぎ対応（Screaming Frog Outlinks想定）
    col_source = pick_col(df, ["Source", "source", "From", "URL Source", "Source URL"])
    col_dest = pick_col(df, ["Destination", "destination", "To", "URL Destination", "Destination URL"])

    if not col_source or not col_dest:
        raise ValueError(
            "CSVに Source/Destination 相当の列が見つかりません。\n"
            f"見つかった列: {list(df.columns)}\n"
            "例: Screaming Frog の Outlinks なら通常は Source / Destination です。"
        )

    # 3) URL正規化（重複ノード削減）
    df[col_source] = df[col_source].astype(str).map(normalize_url)
    df[col_dest] = df[col_dest].astype(str).map(normalize_url)

    # 空URL除外
    df = df[(df[col_source] != "") & (df[col_dest] != "")]

    # 4) 内部リンクに絞る
    df = df[df[col_source].map(lambda x: is_internal(x, TARGET_DOMAIN))]
    df = df[df[col_dest].map(lambda x: is_internal(x, TARGET_DOMAIN))]

    # 5) 追加フィルタ（列がある時だけ適用）
    col_status = pick_col(df, ["Status Code", "Status", "HTTP Status Code", "Response Code"])
    if ONLY_STATUS_200 and col_status:
        # 数値になってる/文字列になってる両方を吸収
        df = df[pd.to_numeric(df[col_status], errors="coerce") == 200]

    col_follow = pick_col(df, ["Follow", "follow", "Is Follow", "Link Follow"])
    if ONLY_FOLLOW_TRUE and col_follow:
        df = df[to_bool_str(df[col_follow])]

    if DROP_SAME_URL_EDGE:
        df = df[df[col_source] != df[col_dest]]

    # 6) グラフ構築（有向）
    edges = df[[col_source, col_dest]].dropna()
    edges = edges.rename(columns={col_source: "Source", col_dest: "Destination"})

    if edges.empty:
        raise ValueError(
            "エッジが0件です。フィルタが厳しすぎる可能性があります。\n"
            "確認ポイント:\n"
            f"- TARGET_DOMAIN={TARGET_DOMAIN} が正しいか\n"
            "- ONLY_STATUS_200 / ONLY_FOLLOW_TRUE を一旦 False にして増えるか\n"
            "- CSVが本当に内部リンク（同一ドメイン）を含んでいるか"
        )

    G = nx.from_pandas_edgelist(edges, source="Source", target="Destination", create_using=nx.DiGraph())

    if G.number_of_nodes() == 0:
        raise ValueError("ノードが0です。CSVの内容かフィルタ条件を見直してください。")

    # 7) PageRank（SciPy不要のpower iteration固定）
    # ※ networkx の環境によっては pagerank() が scipy を要求するケースがあるため、
    #    pagerank_scipy に落ちない形で power iteration を明示します。
    pr = nx.pagerank(G, alpha=0.85, max_iter=200, tol=1.0e-6)
    nx.set_node_attributes(G, pr, "pagerank")

    # 8) 重い場合のノード削減（上位Nのみ表示）
    if TOP_N_NODES is not None and G.number_of_nodes() > TOP_N_NODES:
        top_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:TOP_N_NODES]
        top_nodes_set = {n for n, _ in top_nodes}
        G = G.subgraph(top_nodes_set).copy()
        pr = {n: pr[n] for n in G.nodes()}

    # 9) PyVisでHTML出力
    net = Network(height=HEIGHT, width=WIDTH, directed=True, bgcolor=BG_COLOR, font_color=FONT_COLOR)
    net.barnes_hut()

    # ノード追加（PageRankに応じてサイズ調整）
    max_pr = max(pr.values()) if pr else 1.0
    for n in G.nodes():
        score = (pr.get(n, 0.0) / max_pr) if max_pr else 0.0
        size = 8 + (score * 40)  # 8〜48くらい
        label = n if SHOW_LABEL else ""
        net.add_node(
            n,
            label=label,
            title=f"{n}<br>PageRank: {pr.get(n, 0.0):.8f}",
            size=size
        )

    # エッジ追加
    for s, t in G.edges():
        net.add_edge(s, t)

    # 10) 保存
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(OUTPUT))

    print(f"Saved: {OUTPUT}")
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Input edges(after filters): {len(edges)}")


if __name__ == "__main__":
    main()
