import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import os

#  Load your dataset

file_path = "/Users/dasari/Downloads/Cross_ Domain_Knowledge/ner.csv"

if not os.path.exists(file_path):
    raise FileNotFoundError(f"❌ File not found: {file_path}")

df = pd.read_csv(file_path)
print("✅ File loaded successfully!")
print("Columns in dataset:", list(df.columns))
print(df.head(), "\n")

# Check columns and adapt dynamically
# -------------------------------------
# Your ner.csv has columns: ['Sentence #', 'Sentence', 'POS', 'Tag']

if "Sentence" in df.columns:
    # Example relation extraction (simplified demo)
    # Here we use the 'Sentence' column to create sample triples
    triples = []
    for i, row in df.iterrows():
        sentence = str(row["Sentence"])
        tag = str(row["Tag"])
        # Example: treat sentence as "subject" and tag as "object"
        triples.append({
            "subject": f"Sentence_{i+1}",
            "relation": "has_tag",
            "object": tag
        })
else:
    raise ValueError("❌ Expected 'Sentence' column not found in dataset.")

triples_df = pd.DataFrame(triples)
print("✅ Triples created successfully!\n")
print(triples_df.head(), "\n")

# Create Knowledge Graph
G = nx.DiGraph()

for _, row in triples_df.iterrows():
    subj = row["subject"]
    obj = row["object"]
    rel = row["relation"]
    G.add_edge(subj, obj, label=rel)

print(f"✅ Graph built with {len(G.nodes())} nodes and {len(G.edges())} edges.\n")

# -------------------------------------
# 4️⃣ Static Visualization (Matplotlib)
# -------------------------------------
plt.figure(figsize=(10, 7))
pos = nx.spring_layout(G, k=0.3)
nx.draw(G, pos, with_labels=True, node_size=1200, node_color="lightblue", arrows=True, font_size=8)
nx.draw_networkx_edge_labels(G, pos, edge_labels=nx.get_edge_attributes(G, 'label'), font_color='red')
plt.title("Knowledge Graph (Matplotlib)", fontsize=14)
plt.show()

# Interactive Visualization (PyVis)

net = Network(height="600px", width="100%", directed=True, bgcolor="#222222", font_color="white")
net.from_nx(G)

# Add labeled edges
for e in G.edges(data=True):
    net.add_edge(e[0], e[1], title=e[2]["label"], label=e[2]["label"])

# Save and open interactive visualization
output_html = os.path.join(os.path.dirname(file_path), "knowledge_graph.html")
net.show(output_html)
print(f"🌐 Interactive Knowledge Graph saved as: {output_html}")
