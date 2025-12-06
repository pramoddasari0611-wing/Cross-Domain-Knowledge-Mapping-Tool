# al_knowmap_prototype_fixed.py
import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import tempfile
import os
import warnings
import html
import shutil

# -----------------------
# Configuration / Globals
# -----------------------
DEFAULT_DATASET = "/Users/dasari/Downloads/Cross_ Domain_Knowledge/ner_triples.csv"
# how many top categories to show on charts
DEFAULT_TOP_N = 20
# maximum nodes to render in full pyvis graph (to keep browser responsive)
DEFAULT_MAX_PYVIS_NODES = 200

# -----------------------
# Small utilities
# -----------------------
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip() for c in df.columns]
    # also keep a lowercase mapping for fuzzy lookup
    return df

def find_best_column_names(df: pd.DataFrame):
    """
    Return a mapping for keys: Subject, Relation, Object, Subject_Type, Object_Type
    Accepts a variety of column name variants (case-insensitive).
    """
    cols = {c.lower(): c for c in df.columns}
    def pick(candidates, default=None):
        for cand in candidates:
            if cand.lower() in cols:
                return cols[cand.lower()]
        return default

    mapping = {
        "Subject": pick(["Subject", "subject", "SUBJECT", "subj", "subject_text", "sentence"]),
        "Relation": pick(["Relation", "relation", "RELATION", "rel", "predicate"]),
        "Object": pick(["Object", "object", "OBJECT", "obj"]),
        "Subject_Type": pick(["Subject_Type", "subject_type", "subj_type", "Subject Type"]),
        "Object_Type": pick(["Object_Type", "object_type", "obj_type", "Object Type"]),
    }
    return mapping

def safe_read_csv(path_or_buffer):
    # read flexibly and normalize
    df = pd.read_csv(path_or_buffer)
    df = normalize_columns(df)
    return df

# -----------------------
# Load dataset (auto or uploaded)
# -----------------------
def load_default_dataset():
    if os.path.exists(DEFAULT_DATASET):
        try:
            return safe_read_csv(DEFAULT_DATASET), DEFAULT_DATASET
        except Exception:
            return None, None
    return None, None

# -----------------------
# Build NetworkX graph
# -----------------------
def build_graph(df, colmap):
    G = nx.Graph()
    # add edges
    subj_col = colmap["Subject"]
    obj_col = colmap["Object"]
    rel_col = colmap["Relation"]
    stype_col = colmap.get("Subject_Type")
    otype_col = colmap.get("Object_Type")

    for _, row in df.iterrows():
        # make sure missing values don't break things
        subj = str(row[subj_col]) if pd.notna(row[subj_col]) else ""
        obj  = str(row[obj_col])  if pd.notna(row[obj_col])  else ""
        rel  = str(row[rel_col])  if pd.notna(row[rel_col])  else ""

        if subj == "" or obj == "":
            continue

        s_type = str(row[stype_col]) if stype_col and stype_col in df.columns and pd.notna(row[stype_col]) else "unknown"
        o_type = str(row[otype_col]) if otype_col and otype_col in df.columns and pd.notna(row[otype_col]) else "unknown"

        # avoid empty node names
        G.add_node(subj, type=s_type)
        G.add_node(obj, type=o_type)
        # store relation as edge attribute; use count to weight duplicates
        if G.has_edge(subj, obj):
            # increment weight if same relation
            G[subj][obj].setdefault("weight", 1)
            G[subj][obj]["weight"] += 1
            # keep relation string (if multiple relations, keep concatenated unique)
            prev = G[subj][obj].get("relation", "")
            if rel and rel not in prev.split("|"):
                G[subj][obj]["relation"] = "|".join(filter(None, [prev, rel]))
        else:
            G.add_edge(subj, obj, relation=rel, weight=1)
    return G

# -----------------------
# Draw PyVis graph (returns path to temp html)
# -----------------------
def draw_pyvis_graph(G, max_nodes=DEFAULT_MAX_PYVIS_NODES, node_color_map=None):
    # limit nodes to top N by degree if graph is large
    if G.number_of_nodes() > max_nodes:
        # pick top-degree nodes
        deg = dict(G.degree())
        top_nodes = sorted(deg.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        selected = set([n for n, _ in top_nodes])
        # include edges where both endpoints in selected
        subG = G.subgraph(selected).copy()
    else:
        subG = G

    net = Network(height="700px", width="100%", bgcolor="#0e1117", font_color="white", directed=False)
    net.force_atlas_2based()  # improves layout in many cases

    # default color mapping if none given
    default_color_map = {
        "geo": "#2ca02c",
        "gpe": "#1f77b4",
        "org": "#ff7f0e",
        "tim": "#9467bd",
        "unknown": "#7f7f7f",
    }
    if node_color_map:
        color_map = {**default_color_map, **node_color_map}
    else:
        color_map = default_color_map

    # add nodes
    for node, attrs in subG.nodes(data=True):
        ntype = attrs.get("type", "unknown")
        color = color_map.get(ntype, "#7f7f7f")
        title = f"{html.escape(str(node))}<br>type: {html.escape(str(ntype))}"
        net.add_node(str(node), label=str(node), title=title, color=color)

    # add edges
    for u, v, attrs in subG.edges(data=True):
        title = attrs.get("relation", "")
        weight = attrs.get("weight", 1)
        net.add_edge(str(u), str(v), title=str(title), value=int(weight))

    # write to a temp file
    tmpdir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmpdir, "knowledge_graph.html")
    net.save_graph(tmp_path)
    return tmp_path, tmpdir

# -----------------------
# Statistics plotting helpers (matplotlib)
# -----------------------
# filter matplotlib glyph warnings that you saw earlier
warnings.filterwarnings("ignore", message="Glyph .* missing from font")

def plot_bar_chart_streamlit(series: pd.Series, title: str, top_n=DEFAULT_TOP_N):
    # only top N
    top = series.value_counts().nlargest(top_n)
    fig, ax = plt.subplots(figsize=(6,4))
    top.plot(kind="bar", ax=ax)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Count")
    ax.set_xlabel("")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="AI-KnowMap Prototype", layout="wide")

# Sidebar navigation and controls
with st.sidebar:
    st.markdown("## 🚀 AI-KnowMap Prototype")
    page = st.radio("Navigation", [
        "📥 Upload & Preview",
        "📊 Graph Statistics",
        "🧠 Knowledge Graph",
        "🔍 Query-Based Subgraph",
        "🧹 Cleaning Tools"
    ])
    st.divider()
    st.header("Load Dataset")
    uploaded_file = st.file_uploader("Upload a triples CSV (Subject,Relation,Object,...)", type=["csv"])
    st.write("")  # spacing
    st.caption("Or the app will attempt to auto-load the default dataset.")

    st.divider()
    st.header("Graph options")
    max_nodes = st.slider("Max nodes to render in graph (top by degree)", 20, 2000, DEFAULT_MAX_PYVIS_NODES, step=10)
    top_n_chart = st.slider("Top categories to show in charts", 5, 50, DEFAULT_TOP_N, step=1)
    st.divider()
    st.caption("Node colors are mapped by type (geo,gpe,org,tim,unknown).")

# Load df (uploaded takes precedence)
if uploaded_file is not None:
    try:
        df = safe_read_csv(uploaded_file)
        source_path = "Uploaded file"
        st.sidebar.success("✅ CSV uploaded (used).")
    except Exception as e:
        st.sidebar.error(f"Failed to read uploaded CSV: {e}")
        df = None
        source_path = None
else:
    df, source_path = load_default_dataset()
    if df is not None:
        st.sidebar.info(f"Auto-loaded: {source_path}")
    else:
        st.sidebar.warning("No default dataset found. Please upload.")

# If dataset loaded, attempt to map expected columns
colmap = None
if df is not None:
    colmap = find_best_column_names(df)
    # If any required mapping missing, show a warning and ask user to map columns manually
    missing = [k for k,v in colmap.items() if k in ("Subject","Relation","Object") and not v]
    if missing:
        st.sidebar.error(f"Could not detect required columns automatically: {missing}")
        # allow manual mapping in sidebar
        st.sidebar.markdown("### Manual column mapping")
        for key in ("Subject","Relation","Object","Subject_Type","Object_Type"):
            opt = st.sidebar.selectbox(f"Map {key} to column:", options=[""] + list(df.columns), index=0, key=f"map_{key}")
            if opt:
                colmap[key] = opt
        # re-check
        missing = [k for k in ("Subject","Relation","Object") if not colmap.get(k)]
        if missing:
            st.warning("Required columns still missing — please upload a correct triples CSV (Subject,Relation,Object).")
            df = None  # disable further pages

# -------------------------
# Page: Upload & Preview
# -------------------------
if page == "📥 Upload & Preview":
    st.title("📥 Upload & Preview Dataset")
    if df is None:
        st.info("Please upload a triples CSV in the left sidebar, or place the default file at:")
        st.code(DEFAULT_DATASET)
    else:
        st.markdown(f"**Source:** {source_path}")
        st.markdown(f"**Rows:** {len(df)} — **Columns:** {list(df.columns)}")
        st.dataframe(df.head(200), use_container_width=True)

# -------------------------
# Page: Graph Statistics
# -------------------------
elif page == "📊 Graph Statistics":
    st.title("📊 Graph Statistics")
    if df is None:
        st.warning("No dataset loaded.")
    else:
        st.subheader("Basic Stats")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total records (rows)", len(df))
        col2.metric("Unique Subjects", df[colmap["Subject"]].nunique() if colmap else "N/A")
        col3.metric("Unique Objects", df[colmap["Object"]].nunique() if colmap else "N/A")

        st.divider()
        st.subheader("Distributions (Top categories)")
        c1, c2 = st.columns(2)
        with c1:
            plot_bar_chart_streamlit(df[colmap["Subject"]], "Most frequent Subjects", top_n_chart)
            plot_bar_chart_streamlit(df[colmap["Subject_Type"]] if colmap.get("Subject_Type") in df.columns else pd.Series(["unknown"]*len(df)), "Subject Type distribution", top_n_chart)
        with c2:
            plot_bar_chart_streamlit(df[colmap["Object"]], "Most frequent Objects", top_n_chart)
            plot_bar_chart_streamlit(df[colmap["Object_Type"]] if colmap.get("Object_Type") in df.columns else pd.Series(["unknown"]*len(df)), "Object Type distribution", top_n_chart)

# -------------------------
# Page: Knowledge Graph
# -------------------------
elif page == "🧠 Knowledge Graph":
    st.title("🧠 Knowledge Graph Visualization")
    if df is None:
        st.warning("No dataset loaded.")
    else:
        G = build_graph(df, colmap)
        st.markdown(f"Graph: **{G.number_of_nodes()}** nodes, **{G.number_of_edges()}** edges (will render top `{max_nodes}` nodes by degree).")
        # color map example: you can customize
        color_map = {"geo":"#2ca02c","gpe":"#1f77b4","org":"#ff7f0e","tim":"#9467bd","unknown":"#7f7f7f"}
        html_path, tmpdir = draw_pyvis_graph(G, max_nodes=max_nodes, node_color_map=color_map)

        # embed
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        st.components.v1.html(html, height=750, scrolling=True)

        # cleanup temp directory when user clicks a button (avoid removing while in use)
        if st.button("Remove temporary graph files"):
            try:
                shutil.rmtree(tmpdir)
                st.success("Temporary files removed.")
            except Exception as e:
                st.error(f"Could not remove temp files: {e}")

# -------------------------
# Page: Query-Based Subgraph
# -------------------------
elif page == "🔍 Query-Based Subgraph":
    st.title("🔍 Query-Based Subgraph Explorer")
    if df is None:
        st.warning("No dataset loaded.")
    else:
        G = build_graph(df, colmap)
        node_list = sorted(G.nodes())
        node_choice = st.selectbox("Pick a node:", node_list)
        if node_choice:
            neighbors = list(G.neighbors(node_choice))
            st.write("Connected Nodes (neighbors):")
            st.json(neighbors)

            # show small subgraph (center node + neighbors)
            sub_nodes = [node_choice] + neighbors
            SG = G.subgraph(sub_nodes).copy()
            html_path, tmpdir = draw_pyvis_graph(SG, max_nodes=len(sub_nodes))
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()
            st.components.v1.html(html, height=600, scrolling=True)

# -------------------------
# Page: Cleaning Tools
# -------------------------
elif page == "🧹 Cleaning Tools":
    st.title("🧹 Graph Cleaning Tools (preview)")
    if df is None:
        st.warning("No dataset loaded.")
    else:
        st.info("This section provides helpers for detecting duplicates, low-value relations and orphan nodes (non-exhaustive).")
        G = build_graph(df, colmap)

        # sample duplicate detection by 'node name' exact duplicates (case-insensitive)
        st.subheader("Duplicate node name counts (top 30)")
        node_counts = pd.Series([n.lower() for n in df[colmap["Subject"]].tolist() + df[colmap["Object"]].tolist()]).value_counts()
        st.dataframe(node_counts.head(30))

        st.subheader("Orphan nodes (nodes with degree 0) — sample 50")
        orphans = [n for n, d in G.degree() if d == 0]
        st.write(f"Orphans found: {len(orphans)} — sample:")
        st.write(orphans[:50])

        st.subheader("Low-value relations (sample)")
        # heuristic: relations that connect very common generic tokens (like 'related_to' only) - show top relations
        if "Relation" in colmap and colmap["Relation"] in df.columns:
            rel_counts = df[colmap["Relation"]].value_counts().head(30)
            st.dataframe(rel_counts)
        else:
            st.write("No relation column available to analyze.")

        st.markdown("---")
        st.caption("Cleaning tools are demonstration-only. For merging/removing nodes programmatically, incorporate your business rules and persist changes.")

# End of file
