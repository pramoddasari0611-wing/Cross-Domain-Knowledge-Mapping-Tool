import streamlit as st
import pandas as pd
import streamlit.components.v1 as components 
import networkx as nx                    
from pyvis.network import Network         
import os
import plotly.express as px

# Page Configuration

st.set_page_config(page_title="Admin Dashboard", layout="wide")

#  Custom Dark Theme CSS

st.markdown("""
    <style>
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #1e293b;  /* dark gray-blue */
        color: white;
        padding: 1.5rem 1rem;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] p {
        color: white !important;
    }

    /* Main background */
    .main {
        background-color: #0f172a;
        color: white;
        padding: 2rem;
        border-radius: 10px;
    }

    /* Metric cards text */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #10b981; /* teal accent */
    }

    table {
        color: white !important;
    }

    /* Header styling */
    h1, h2, h3 {
        color: white !important;
    }

    </style>
""", unsafe_allow_html=True)

# Sidebar Navigation

st.sidebar.title("Admin Controls")
view = st.sidebar.selectbox(
    "Select a View:",
    ["Overview", "Entity Viewer", "Relation Graph", "Feedback Panel"]
)

# Load Dataset

file_paths = [
    "/Users/dasari/Downloads/Cross_ Domain_Knowledge/ner_triples.csv",
    "/Users/dasari/Downloads/Cross_ Domain_Knowledge/data/processed/ner_triples.csv",
    "/Users/dasari/Downloads/Cross_ Domain_Knowledge/data/processed/triples.csv"
]

data = None
for path in file_paths:
    try:
        data = pd.read_csv(path)
        
        # --- CRITICAL: Ensure correct column names for Graph visualization ---
        if data.shape[1] < 3:
             st.sidebar.warning(f"File at {path} has fewer than 3 columns. Skipping.")
             continue
        
        # Assume the first three columns are Subject, Relation, Object
        data.columns = [f'col_{i}' for i in range(data.shape[1])]
        data = data.rename(columns={'col_0': 'subject', 'col_1': 'relation', 'col_2': 'object'})
        
        st.sidebar.success(f"Loaded dataset from:\n{path}")
        break
    except Exception as e:
        continue

if data is None:
    st.error(" No valid dataset found. Please check your file paths.")
    st.stop()


#  Overview Section 

if view == "Overview":
    st.title(" Admin Dashboard & Feedback Panel")
    st.markdown("A control center for monitoring and improving your semantic knowledge graph.")

    # Define the configuration dictionary for Plotly charts
    # 'width': 'stretch' handles the old use_container_width=True
    plotly_config = {'displayModeBar': False, 'responsive': True, 'width': 'stretch'}
    
    # --- 1. Graph Statistics ---
    st.header(" Graph Statistics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Records", len(data))
    
    if 'subject' in data.columns and 'object' in data.columns:
        all_entities = pd.concat([data['subject'], data['object']]).unique()
        col2.metric("Total Unique Entities", len(all_entities))
    else:
        col2.metric("Columns / Attributes", data.shape[1])
        
    col3.metric("Missing Values", data.isna().sum().sum())
    
    st.markdown("---")

    # --- 2. Distribution Analysis (Statistics Graphs) ---
    if 'subject' in data.columns and 'relation' in data.columns and 'object' in data.columns:
        st.header("📦 Distribution Analysis")
        
        entity_counts = pd.concat([data['subject'], data['object']]).value_counts().head(10)
        relation_counts = data['relation'].value_counts().head(10)

        colA, colB = st.columns(2)
        
        with colA:
            st.subheader("Top 10 Most Frequent Entities")
            if not entity_counts.empty:
                fig_entity = px.bar(
                    entity_counts, 
                    x=entity_counts.values, 
                    y=entity_counts.index, 
                    orientation='h',
                    title='Entity Frequency (Subject & Object)',
                    labels={'x': 'Count', 'y': 'Entity'},
                    color_discrete_sequence=['#10b981']
                )
                fig_entity.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    plot_bgcolor='#1e293b', 
                    paper_bgcolor='#0f172a',
                    font_color='white'
                )
                # FIX: Using config dictionary
                st.plotly_chart(fig_entity, config=plotly_config) 
            else:
                st.info("No entities found for analysis.")
                
        with colB:
            st.subheader("Top 10 Most Frequent Relations")
            if not relation_counts.empty:
                fig_relation = px.bar(
                    relation_counts, 
                    x=relation_counts.values, 
                    y=relation_counts.index, 
                    orientation='h',
                    title='Relation Type Frequency',
                    labels={'x': 'Count', 'y': 'Relation'},
                    color_discrete_sequence=['#ef4444']
                )
                fig_relation.update_layout(
                    yaxis={'categoryorder':'total ascending'},
                    plot_bgcolor='#1e293b', 
                    paper_bgcolor='#0f172a',
                    font_color='white'
                )
                # FIX: Using config dictionary
                st.plotly_chart(fig_relation, config=plotly_config) 
            else:
                st.info("No relations found for analysis.")

    st.markdown("---")
    
    # --- 3. Sample Data ---
    st.header("Sample of Extracted Relations")
    st.dataframe(data.head(10))


#  Entity Viewer Section

elif view == "Entity Viewer":
    st.title(" Explore Extracted Relations")

    if 'subject' in data.columns and 'relation' in data.columns and 'object' in data.columns:
        
        search_column = st.selectbox(
            "Select the column to search:",
            ['All Columns (Subject, Relation, Object)', 'Subject', 'Relation', 'Object']
        )
        
        search_query = st.text_input("Enter search query (e.g., 'related_to' or 'London'):", "")
        
        if search_query:
            if search_column == 'All Columns (Subject, Relation, Object)':
                filtered_data = data[data[['subject', 'relation', 'object']].apply(
                    lambda row: row.astype(str).str.contains(search_query, case=False).any(), axis=1)]
            
            else:
                column_name = search_column.lower()
                filtered_data = data[
                    data[column_name].astype(str).str.contains(search_query, case=False)
                ]
        else:
            filtered_data = data
            
    else:
        st.warning("Data columns 'subject', 'relation', 'object' not found. Displaying full dataset.")
        filtered_data = data
    
    num_rows = st.slider("Number of rows to display:", 10, min(len(filtered_data), 1000), 20)
    st.dataframe(filtered_data.head(num_rows))
    st.caption(f"Showing {min(num_rows, len(filtered_data))} of {len(filtered_data)} matching records.")

# Relation Graph Section

elif view == "Relation Graph":
    st.title(" Knowledge Relation Graph")
    st.markdown("Use the controls below to **filter the graph** by Entity or Relation Type, or **reduce the sample size**.")
    
    # --- Controls for Filtering and Limiting ---
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        filter_entity = st.text_input("Filter by Entity (Subject or Object):", "")
    with col_filter2:
        filter_relation = st.text_input("Filter by Relation Type (e.g., 'located_in'):", "")
        
    GRAPH_ROW_LIMIT = st.slider("Maximum number of rows (edges) to sample:", 10, min(len(data), 5000), 500)
    st.markdown("---")
    
    if 'subject' not in data.columns or 'object' not in data.columns or 'relation' not in data.columns:
        st.error("Cannot display graph: The loaded dataset must contain 'subject', 'relation', and 'object' columns.")
        
    else:
        df_display = data.copy()
        
        # 1. Apply Entity Filter
        if filter_entity:
            df_display = df_display[
                (df_display['subject'].astype(str).str.contains(filter_entity, case=False)) |
                (df_display['object'].astype(str).str.contains(filter_entity, case=False))
            ]
        
        # 2. Apply Relation Filter
        if filter_relation:
             df_display = df_display[
                df_display['relation'].astype(str).str.contains(filter_relation, case=False)
            ]
        
        if df_display.empty:
            st.warning("No records found for the given filter combination.")
        else:
            # 3. Apply Row Limit
            df_sample = df_display.head(GRAPH_ROW_LIMIT)
            
            # 4. Create NetworkX Graph
            G = nx.from_pandas_edgelist(
                df_sample,
                source='subject', 
                target='object', 
                edge_attr='relation',
                create_using=nx.DiGraph() 
            )
            
            # 5. Calculate Degree Centrality (for node size)
            degrees = dict(G.degree) 
            max_degree = max(degrees.values()) if degrees else 1
            
            def scale_size(degree):
                if max_degree == 0: return 10
                return 10 + 40 * (degree / max_degree) 

            # 6. Initialize Pyvis Network
            net = Network(height="750px", width="100%", bgcolor="#0f172a", font_color="white", notebook=True)
            net.toggle_physics(True)
            
            # Transfer nodes and add size/title/label
            for node in G.nodes:
                degree = degrees.get(node, 0)
                net.add_node(
                    n_id=node, 
                    label=node, 
                    title=f"Connections: {degree}", 
                    size=scale_size(degree),       
                    color="#10b981"                
                )
                
            # Transfer edges
            for u, v, data in G.edges(data=True):
                net.add_edge(u, v, title=data['relation'], label=data['relation'], color="#475569")

            # 7. Render Graph
            try:
                path = 'pyvis_temp_graph.html'
                net.set_options("""
                    var options = {
                      "physics": {
                        "barnesHut": {
                          "gravitationalConstant": -8000,
                          "springLength": 150,
                          "springConstant": 0.001
                        },
                        "minVelocity": 0.75
                      }
                    }
                """)
                net.save_graph(path)
                
                HtmlFile = open(path, 'r', encoding='utf-8')
                components.html(HtmlFile.read(), height=770, scrolling=True)
                st.caption(f"Showing a graph built from the top **{len(df_sample)}** relevant records. Total unique nodes: **{len(G.nodes)}**.")
                os.remove(path) 
            except Exception as e:
                st.error(f"Failed to render interactive graph. Error: {e}")

# Feedback Section

elif view == "Feedback Panel":
    st.title(" User Feedback Summary")

    st.markdown("""
    This section will later store user feedback about incorrect relations or suggestions for merging nodes.
    It helps improve your semantic graph over time.
    """)

    feedback_placeholder = st.text_area(" Enter feedback about incorrect or missing relations:")
    if st.button("Submit Feedback"):
        st.success("Feedback submitted successfully!")

# Footer

st.markdown("---")
st.markdown("**Developed for Semantic Knowledge Graph Monitoring — Cross Domain Project.**")