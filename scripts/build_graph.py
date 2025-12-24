import pandas as pd
import networkx as nx
from pyvis.network import Network
from pathlib import Path

INPUT = Path("data/links.csv")
OUTPUT = Path("docs/index.html")

df = pd.read_csv(INPUT)

# 有向グラフ（内部リンクは方向が重要）
G = nx.from_pandas_edgelist(df, source="source", target="target", create_using=nx.DiGraph())

# 重要度の例：PageRank（リンクジュースの偏り可視化に便利）
pr = nx.pagerank(G, alpha=0.85)
nx.set_node_attributes(G, pr, "pagerank")

# PyVisで可視化（HTML）
net = Network(height="800px", width="100%", directed=True, bgcolor="#0b0f19", font_color="#ffffff")
net.barnes_hut()

for n in G.nodes():
    size = 10 + (pr.get(n, 0) * 300)  # 重要ページほど大きく
    net.add_node(n, label="", title=n, size=size)

for s, t in G.edges():
    net.add_edge(s, t)

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
net.show(str(OUTPUT))

print(f"Saved: {OUTPUT}")
