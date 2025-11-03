import pandas as pd
import networkx as nx
from pyvis.network import Network
import os

# Locate your triples CSV file

possible_files = [
    "/Users/dasari/Downloads/Cross_ Domain_Knowledge/data/processed/ner_triples.csv",
    "/Users/dasari/Downloads/Cross_ Domain_Knowledge/ner_triples.csv",
    "/Users/dasari/Downloads/Cross_ Domain_Knowledge/data/processed/triples.csv"
]

triples_path = next((f for f in possible_files if os.path.exists(f)), None)
if not triples_path:
    raise FileNotFoundError("❌ No triples CSV found. Please check your file path.")

print(f" Using file: {triples_path}")

# Load and check data

df = pd.read_csv(triples_path)
df.columns = df.columns.str.lower()
print("Columns in dataset:", df.columns.tolist())

required_cols = {"subject", "relation", "object"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"❌ Missing one of {required_cols} in dataset!")

# Limit to a smaller subset for visualization (10–20 nodes)

sample_size = 20
if len(df) > sample_size:
    df = df.sample(n=sample_size, random_state=42)

print(f" Using {len(df)} triples for visualization")

# Build a directed graph

G = nx.DiGraph()
for _, row in df.iterrows():
    subj, rel, obj = str(row["subject"]), str(row["relation"]), str(row["object"])
    G.add_node(subj, color="lightblue")
    G.add_node(obj, color="lightgreen")
    G.add_edge(subj, obj, label=rel)

print(f" Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.")


# Create PyVis Network

net = Network(
    height="750px",
    width="100%",
    bgcolor="#202020",
    font_color="white",
    directed=True
)

net.from_nx(G)

# Add relation labels on edges
for e in G.edges(data=True):
    rel = e[2].get("label", "")
    net.add_edge(e[0], e[1], title=rel, label=rel)

# Enable physics for layout
net.toggle_physics(True)

# Save to HTML

output_file = "/Users/dasari/Downloads/Cross_ Domain_Knowledge/my_knowledge_graph.html"
net.write_html(output_file)

print(f"\n🌐 Graph saved to:\n{output_file}")
print("👉 Open it manually in Chrome or Safari (not MKPlayer).")
