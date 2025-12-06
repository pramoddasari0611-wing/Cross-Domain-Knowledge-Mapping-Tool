# al_knowmap_app.py
# Al-KnowMap prototype - integrated Streamlit app
# Features:
# - Upload or auto-detect triples CSV
# - View dataset & extracted entities/relations
# - Generate natural-language sentences from triples
# - Semantic search (SentenceTransformers)
# - Build & visualize a subgraph (NetworkX + PyVis)
# - Save feedback & export triples

import streamlit as st
import pandas as pd
import os
import networkx as nx
from pyvis.network import Network
import tempfile
from sentence_transformers import SentenceTransformer, util

# Config / Paths

st.set_page_config(page_title="Al-KnowMap Prototype", layout="wide")

DEFAULT_PATHS = [
    "/Users/dasari/Downloads/Cross_ Domain_Knowledge/data/processed/ner_triples.csv",
    "/Users/dasari/Downloads/Cross_ Domain_Knowledge/ner_triples.csv",
    "/Users/dasari/Downloads/Cross_ Domain_Knowledge/data/processed/triples.csv",
    "/Users/dasari/Downloads/Cross_ Domain_Knowledge/ner.csv"
]

# Helpers

@st.cache_data
def load_df_from_path(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df, p
            except Exception:
                continue
    return None, None

def safe_load_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0)
        return pd.read_csv(uploaded_file, encoding="latin1", errors="replace")

def relation_to_sentence(subj, rel, obj):
    """Try to make relation readable. Add mappings as needed."""
    r = str(rel).lower().replace("_", " ")
    mapping = {
        "part of": f"{subj} is part of {obj}.",
        "founded": f"{subj} founded {obj}.",
        "founded by": f"{obj} founded {subj}.",
        "born in": f"{subj} was born in {obj}.",
        "located in": f"{subj} is located in {obj}.",
        "capital of": f"{subj} is the capital of {obj}.",
        "member of": f"{subj} is a member of {obj}.",
        "related to": f"{subj} is related to {obj}.",
    }
    for k in mapping:
        if k in r:
            return mapping[k]
    return f"{subj} {r} {obj}."

@st.cache_resource
def get_sentence_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

def build_graph_from_df(df):
    if {'Subject', 'Relation', 'Object'}.issubset(df.columns):
        G = nx.from_pandas_edgelist(df, 'Subject', 'Object', edge_attr='Relation', create_using=nx.DiGraph())
    else:
        # Try common alternative names
        cols = [c.lower() for c in df.columns]
        subj_col = next((c for c in df.columns if c.lower() in ('subject','entity1','head')), None)
        rel_col  = next((c for c in df.columns if c.lower() in ('relation','rel','predicate')), None)
        obj_col  = next((c for c in df.columns if c.lower() in ('object','entity2','tail')), None)
        if subj_col and rel_col and obj_col:
            G = nx.from_pandas_edgelist(df, subj_col, obj_col, edge_attr=rel_col, create_using=nx.DiGraph())
        else:
            G = nx.DiGraph()
    return G

def render_pyvis_graph(G, height="600px", width="100%"):
    net = Network(height=height, width=width, directed=True, notebook=False)
    net.from_nx(G)
    # show edge labels if present
    for u, v, data in G.edges(data=True):
        label = data.get("Relation") or data.get("relation") or data.get("label") or ""
        if label:
            net.add_edge(u, v, title=str(label), label=str(label))
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)
    return tmp.name

# App UI

st.title("Al-KnowMap — Integrated Prototype")
st.markdown("Upload a triples CSV or let the app auto-detect the processed triples. "
            "Then explore, search semantically, visualize the graph, and collect feedback.")

# Left column: file upload + basic actions
with st.sidebar:
    st.header("Data / Actions")
    uploaded = st.file_uploader("Upload triples CSV (Subject,Relation,Object) or raw NER CSV", type=["csv"])
    if uploaded:
        df = safe_load_csv(uploaded)
        source = "uploaded_file"
        st.success("Uploaded file loaded")
    else:
        df, source = load_df_from_path(DEFAULT_PATHS)
        if df is None:
            st.info("No default file found. Upload or drop your CSV file here.")
        else:
            st.info(f" Auto-loaded: {source}")

    refresh = st.button("Reload detected dataset")
    if refresh and uploaded is None:
        df, source = load_df_from_path(DEFAULT_PATHS)
        if df is not None:
            st.success(f"Reloaded: {source}")
        else:
            st.warning("No dataset found in default paths.")

    st.markdown("---")
    st.write("Model & Search")
    model_load = st.checkbox("Load semantic model (SentenceTransformer)", value=True)
    st.write("Feedback storage:")
    save_feedback_to = st.selectbox("Save feedback to:", ["feedback_log.txt", "user_feedback.csv"])

# If no df loaded stop here
if df is None:
    st.stop()

# Main area: dataset view
st.header("1) Uploaded / Loaded Dataset")
st.write(f"**Source:** {source}")
st.write(f"Rows: {len(df):,} — Columns: {list(df.columns)}")
st.dataframe(df.head(50))

# If dataset contains triples show the relations table
st.header("2) Entities / Relations Table")
# try to coerce expected columns
if {'Subject','Relation','Object'}.issubset(df.columns):
    triples_df = df[['Subject','Relation','Object']].copy()
else:
    # try alternative names
    cols = [c.lower() for c in df.columns]
    subj = next((c for c in df.columns if c.lower() in ('subject','entity1','head')), None)
    rel  = next((c for c in df.columns if c.lower() in ('relation','rel','predicate')), None)
    obj  = next((c for c in df.columns if c.lower() in ('object','entity2','tail')), None)
    if subj and rel and obj:
        triples_df = df[[subj, rel, obj]].copy()
        triples_df.columns = ['Subject','Relation','Object']
    else:
        # Try to guess: if there is a single 'Tag' column with lists it's not easy to present as triples
        st.warning("Could not find standard triples columns (Subject, Relation, Object). Showing full dataset instead.")
        triples_df = df.copy()

st.dataframe(triples_df.head(200))

# 3) Natural-language sentences (take sample or entire)
st.header("3) Natural-language Sentences (from triples)")
sample_n = st.slider("Number of sentences to generate", min_value=10, max_value=min(200, len(triples_df)), value=20)
sentences = []
for _, row in triples_df.head(sample_n).iterrows():
    subj = str(row['Subject'])
    rel = str(row['Relation'])
    obj = str(row['Object'])
    sentences.append(relation_to_sentence(subj, rel, obj))
st.write("Example sentences:")
for s in sentences[:min(20, len(sentences))]:
    st.write("•", s)

# 4) Semantic search
st.header("4) Semantic Search")
if model_load:
    model = get_sentence_model()
    st.info("Embedding sentences...")
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    query = st.text_input("Enter a semantic query (e.g. 'Who is part of a political party?')", "")
    top_k = st.slider("Top K results", 1, 10, 3)
    if query:
        q_emb = model.encode(query, convert_to_tensor=True)
        hits = util.semantic_search(q_emb, sentence_embeddings, top_k=top_k)[0]
        st.write(f"Top {top_k} matches for: **{query}**")
        for h in hits:
            s = sentences[h['corpus_id']]
            st.write(f"- {s} (score: {h['score']:.3f})")
else:
    st.info("Uncheck 'Load semantic model' to skip model loading.")

# 5) Build & visualize graph (subset)
st.header("5) Knowledge Graph Visualization (subset)")
G = build_graph_from_df(triples_df)

if G.number_of_nodes() == 0:
    st.warning("Graph is empty (couldn't find edges). Ensure Subject/Relation/Object exist.")
else:
    st.write(f"Graph nodes: {G.number_of_nodes():,} — edges: {G.number_of_edges():,}")

    # allow user to choose a node to show its ego-network
    node_list = list(G.nodes())[:1000]
    node_choice = st.selectbox("Select a node to view a subgraph (or choose 'None')", options=["None"] + node_list)
    radius = st.slider("Subgraph radius (neighborhood)", 1, 3, 1)

    if node_choice != "None":
        # build ego graph
        ego = nx.ego_graph(G, node_choice, radius=radius, undirected=False)
        st.write(f"Showing subgraph around **{node_choice}**: nodes {len(ego.nodes())}, edges {len(ego.edges())}")
        html_path = render_pyvis_graph(ego, height="600px")
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        st.components.v1.html(html, height=650, scrolling=True)
    else:
        # show small random subset for global view for performance
        sub_n = st.slider("Number of random triples to visualize (for global overview)", 10, 200, 40)
        sub_df = triples_df.sample(min(sub_n, len(triples_df)))
        G_sub = build_graph_from_df(sub_df)
        html_path = render_pyvis_graph(G_sub, height="600px")
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
        st.components.v1.html(html, height=650, scrolling=True)

# 6) Feedback capture (save)
st.header("6) Feedback / Corrections")
fb_text = st.text_area("Report incorrect relation or suggest improvement (describe entity and relation):")
if st.button("Save Feedback"):
    if fb_text.strip():
        path = save_feedback_to if 'save_feedback_to' in globals() else "feedback_log.txt"
        # update path chosen in sidebar variable
        path = st.session_state.get('save_feedback_to', "feedback_log.txt")
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{fb_text.strip()}\n")
        st.success(f"Saved feedback to {path}")
    else:
        st.warning("Please enter feedback before saving.")

# 7) Export / download triples
st.header("7) Export & Save")
if st.button("Export triples (CSV)"):
    out_path = os.path.join(os.getcwd(), "exported_triples.csv")
    triples_df.to_csv(out_path, index=False)
    st.success(f"Triples exported to {out_path}")

st.markdown("---")
st.caption("Al-KnowMap prototype — integrated upload, extract, visualize, search, and feedback.")
