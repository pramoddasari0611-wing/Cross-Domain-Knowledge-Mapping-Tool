import pandas as pd
import networkx as nx
from pyvis.network import Network
import os

# -------------------------------------
# 1️⃣ Load your triples dataset
# -------------------------------------
data_dir = "/Users/dasari/Downloads/Cross_ Domain_Knowledge"
triples_path = os.path.join(data_dir, "ner_triples.csv")

if not os.path.exists(triples_path):
    raise FileNotFoundError(f"❌ File not found: {triples_path}")

triples = pd.read_csv(triples_path)
print("✅ File loaded successfully!")
print("Columns in dataset:", list(triples.columns))
print(triples.head())

# -------------------------------------
# 2️⃣ Create a directed graph
# -------------------------------------
G = nx.DiGraph()

# -------------------------------------
# 3️⃣ Add nodes & edges with types and relations
# -------------------------------------
for _, row in triples.iterrows():
    subj = str(row["Subject"])
    obj = str(row["Object"])
    rel = str(row["Relation"])
    subj_type = str(row.get("Subject_Type", "Entity"))
    obj_type = str(row.get("Object_Type", "Entity"))

    # Color code nodes by type
    color_map = {
        "geo": "#7FDBFF",     # light blue
        "gpe": "#2ECC40",     # green
        "org": "#FFDC00",     # yellow
        "per": "#FF4136",     # red
        "Entity": "#AAAAAA"   # default gray
    }

    subj_color = color_map.get(subj_type.lower(), "#AAAAAA")
    obj_color = color_map.get(obj_type.lower(), "#AAAAAA")

    # Add nodes with color and hover info
    G.add_node(subj, color=subj_color, type=subj_type, title=f"{subj} ({subj_type})")
    G.add_node(obj, color=obj_color, type=obj_type, title=f"{obj} ({obj_type})")

    # Add edge with label
    G.add_edge(subj, obj, label=rel, title=f"{subj} → {rel} → {obj}")

print(f"✅ Graph built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

# -------------------------------------
# 4️⃣ Visualize graph using PyVis
# -------------------------------------
net = Network(height="700px", width="100%", bgcolor="#222222", font_color="white", directed=True)

# Load NetworkX graph
net.from_nx(G)

# ✅ Ensure labels and tooltips show properly
for source, target, data in G.edges(data=True):
    rel = data.get("label", "related_to")
    net.add_edge(source, target, label=rel, title=f"{source} → {rel} → {target}")

for node, data in G.nodes(data=True):
    net.add_node(node, title=data.get("title", node), color=data.get("color", "#AAAAAA"))

# Enable smooth physics (layout)
net.toggle_physics(True)

# -------------------------------------
# 5️⃣ Save and open the graph
# -------------------------------------
output_path = os.path.join(data_dir, "my_knowledge_graph.html")
net.show(output_path)

print(f"🌐 Interactive graph saved at: {output_path}")
