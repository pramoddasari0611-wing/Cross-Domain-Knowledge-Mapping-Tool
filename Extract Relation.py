import streamlit as st
import pandas as pd
import streamlit.components.v1 as components 
import networkx as nx                    
from pyvis.network import Network         
import os
import plotly.express as px

# --- Configuration & Credentials ---
VALID_USERNAME = "admin"
VALID_PASSWORD = "password123"
plotly_config = {'displayModeBar': False, 'responsive': True} 

#  Authentication Functions
def login_page():
    """Displays the login form and handles authentication."""
    # Change page title to AI-KnowMap
    st.set_page_config(page_title="AI-KnowMap Login", layout="centered")
    
    # Custom CSS for Centered Login Box and Dark Theme
    st.markdown("""
    <style>
        .main { background-color: #0f172a; color: white; }
        .login-box {
            background-color: #1e293b; 
            padding: 30px;
            border-radius: 10px;
            box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.5);
            max-width: 400px;
            margin: auto;
            margin-top: 15vh;
        }
        h2 { color: #10b981 !important; text-align: center; margin-bottom: 20px; }
    </style>
    <div class="login-box">
    <h2> AI-KnowMap Login</h2>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username:", key="username_input")
        password = st.text_input("Password:", type="password", key="password_input")
        
        # Extra Extension Field
        st.selectbox("Select Environment:", ["Production", "Staging", "Development"], key="env_select")
        
        st.markdown("---")
        submitted = st.form_submit_button("Log In")

    if submitted:
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.success("Login Successful!")
            st.rerun() 
        else:
            st.error("Invalid username or password.")
            
    st.markdown("</div>", unsafe_allow_html=True)
# CRITICAL FIX for performance: Add Streamlit caching
@st.cache_data
def load_data():
    """Handles data loading with corrected relative paths for Docker and caching."""
    
    st.set_page_config(page_title="Admin Dashboard", layout="wide")
    
    # --- 2. Load the main triples data ---
    try:
        # Use relative read path for Docker
        data = pd.read_csv("ner_triples.csv") 
        
        # Ensure correct column names 
        if data.shape[1] >= 3:
            data.columns = [f'col_{i}' for i in range(data.shape[1])]
            data = data.rename(columns={'col_0': 'subject', 'col_1': 'relation', 'col_2': 'object'})
        
        # The cache stores this result, speeding up app re-runs!
        return data

    except FileNotFoundError:
        st.error(" Required file 'ner_triples.csv' not found. Ensure it was created.")
        st.stop()
    except Exception as e:
        st.error(f" An error occurred during final data loading: {e}")
        st.stop()

#  Main Dashboard Structure
def admin_dashboard(data):
    
    # --- Custom Dark Theme CSS ---
    st.markdown("""
        <style>
        [data-testid="stSidebar"] { background-color: #1e293b; color: white; padding: 1.5rem 1rem; }
        .main { background-color: #0f172a; color: white; padding: 2rem; border-radius: 10px; }
        div[data-testid="stMetricValue"] { font-size: 2rem; color: #10b981; }
        h1, h2, h3 { color: white !important; }
        </style>
    """, unsafe_allow_html=True)
    
    # --- Sidebar Navigation ---
    st.sidebar.title(" Admin Controls")
    st.sidebar.caption(f"User: {st.session_state['username']}")
    
    if st.sidebar.button(" Log Out"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.rerun()

    view = st.sidebar.selectbox(
        "Select a View:",
        ["Overview", "Entity Viewer", "Relation Graph", "Feedback Panel"]
    )
    
    # --- Dataset Upload Status (for visual representation) ---
    st.sidebar.markdown("---") # Separator
    st.sidebar.markdown("### Loaded Dataset")
    st.sidebar.success(" Loaded dataset from:")
    st.sidebar.markdown(f"```\n/usr/src/app/ner_triples.csv\n```") 

    # Check for required columns
    required_cols = ['subject', 'relation', 'object']
    if not all(col in data.columns for col in required_cols):
        st.error(f"Data is missing required columns: {required_cols}. Cannot proceed.")
        return

    # --- Overview Section ---
    if view == "Overview":
        st.title("Admin Dashboard & Distribution Analysis")
        
        # Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", len(data))
        all_entities = pd.concat([data['subject'], data['object']]).unique()
        col2.metric("Total Unique Entities", len(all_entities))
        col3.metric("Total Relations", data['relation'].nunique())
        
        st.markdown("---")

        st.header(" Distribution Analysis")
        entity_counts = pd.concat([data['subject'], data['object']]).value_counts().head(10)
        relation_counts = data['relation'].value_counts().head(10)

        colA, colB = st.columns(2)
        
        with colA:
            st.subheader("Top 10 Most Frequent Entities")
            if not entity_counts.empty:
                fig_entity = px.bar(
                    entity_counts, x=entity_counts.values, y=entity_counts.index, orientation='h',
                    title='Entity Frequency', color_discrete_sequence=['#10b981']
                )
                fig_entity.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='#1e293b', paper_bgcolor='#0f172a', font_color='white')
                st.plotly_chart(fig_entity, config=plotly_config, width='stretch') 
        
        with colB:
            st.subheader("Top 10 Most Frequent Relations")
            if not relation_counts.empty:
                fig_relation = px.bar(
                    relation_counts, x=relation_counts.values, y=relation_counts.index, orientation='h',
                    title='Relation Type Frequency', color_discrete_sequence=['#ef4444']
                )
                fig_relation.update_layout(yaxis={'categoryorder':'total ascending'}, plot_bgcolor='#1e293b', paper_bgcolor='#0f172a', font_color='white')
                st.plotly_chart(fig_relation, config=plotly_config, width='stretch') 

    # --- Entity Viewer Section ---
    elif view == "Entity Viewer":
        st.title("Explore Extracted Relations")
        
        # --- Advanced Filtering for Entity Viewer ---
        search_column = st.selectbox(
            "Select the column to search:",
            ["All Columns (Subject, Relation, Object)", "subject", "relation", "object"]
        )
        search_query = st.text_input("Enter search query (e.g., 'related_to' or 'London'):")
        
        num_rows = st.slider("Number of rows to display:", min_value=10, max_value=min(1000, len(data)), value=20)
        
        filtered_display_data = data.copy()

        if search_query:
            if search_column == "All Columns (Subject, Relation, Object)":
                filtered_display_data = filtered_display_data[
                    filtered_display_data['subject'].str.contains(search_query, case=False, na=False) |
                    filtered_display_data['relation'].str.contains(search_query, case=False, na=False) |
                    filtered_display_data['object'].str.contains(search_query, case=False, na=False)
                ]
            else:
                filtered_display_data = filtered_display_data[
                    filtered_display_data[search_column].str.contains(search_query, case=False, na=False)
                ]

        display_data_final = filtered_display_data[['subject', 'relation', 'object']]
        
        st.dataframe(display_data_final.head(num_rows), width='stretch')


    # --- Relation Graph Section ---
    elif view == "Relation Graph":
        st.title("Knowledge Relation Graph")
        
        # --- Filtering Controls ---
        st.markdown("Use the controls below to filter the graph by Entity or Relation Type, or reduce the sample size.")
        col_filter_1, col_filter_2 = st.columns(2)
        
        with col_filter_1:
            entity_filter = st.text_input("Filter by Entity (Subject or Object):")
        with col_filter_2:
            relation_filter = st.text_input("Filter by Relation Type (e.g., 'located_in'):")

        max_rows = st.slider("Maximum number of rows (edges) to sample:", min_value=10, max_value=min(5000, len(data)), value=100)

        # Apply Filters before limiting rows
        filtered_data = data.copy()
        
        if entity_filter:
            filtered_data = filtered_data[
                filtered_data['subject'].str.contains(entity_filter, case=False, na=False) |
                filtered_data['object'].str.contains(entity_filter, case=False, na=False)
            ]
            
        if relation_filter:
            filtered_data = filtered_data[
                filtered_data['relation'].str.contains(relation_filter, case=False, na=False)
            ]

        # Apply row limit AFTER filtering
        filtered_data = filtered_data.head(max_rows)
        
        # --- PYVIS RENDERING WITH COLOR AND PHYSICS FIX ---
        if not filtered_data.empty:
            
            # Initialize Pyvis network 
            net = Network(height='700px', width='100%', bgcolor='#222222', font_color='white', directed=True)
            
            # Add nodes and edges
            for index, row in filtered_data.iterrows():
                net.add_node(row['subject'], title=row['subject'], color='#10b981', size=15)
                net.add_node(row['object'], title=row['object'], color='#10b981', size=15)
                net.add_edge(
                    row['subject'], 
                    row['object'], 
                    title=row['relation'], 
                    label=row['relation'], 
                    color='#CCCCCC', 
                    arrows='to'
                )

            # Apply physics for smooth layout
            net.toggle_physics(True) 
            net.set_options("""
            var options = {
              "physics": {
                "solver": "forceAtlas2Based",
                "forceAtlas2Based": {
                  "gravitationalConstant": -100,
                  "springLength": 100,
                  "springConstant": 0.08
                },
                "minVelocity": 0.75,
                "timestep": 0.5
              }
            }
            """)

            # Save and display the graph HTML
            try:
                html_path = 'my_knowledge_graph.html'
                net.save_graph(html_path) 
                
                with open(html_path, 'r', encoding='utf-8') as f:
                    html_code = f.read()
                
                components.html(html_code, height=750)
            except Exception as e:
                st.error(f"Error rendering graph: {e}")
        else:
            st.warning("No relations found based on the current filters.")


    # --- Feedback Section ---
    elif view == "Feedback Panel":
        st.title("User Feedback Summary")
        st.markdown("This panel is used for logging and improving the graph.")
        feedback_placeholder = st.text_area(" Enter feedback about incorrect or missing relations:")
        if st.button("Submit Feedback"):
            st.success(" Feedback submitted successfully!")

#  Application Entry Point
def main():
    # Initialize session state for authentication
    if "authenticated" not in st.session_state:
        st.session_state["authenticated"] = False
        st.session_state["username"] = None

    if st.session_state["authenticated"]:
        # Load data is now cached and runs faster!
        data = load_data()
        admin_dashboard(data)
    else:
        login_page()

if __name__ == "__main__":
    main()