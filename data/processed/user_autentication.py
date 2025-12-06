import streamlit as st

# --- 1. Login Page Function ---
def login_page():
    """Displays the login form and handles authentication."""
    
    # Simple hardcoded credentials for demonstration
    # In a real app, these would come from a database or config file
    VALID_USERNAME = "admin"
    VALID_PASSWORD = "password123"
    
    st.set_page_config(page_title="AI-KNOWMAP", layout="centered")
    
    # Custom CSS for Centered Login Box (optional)
    st.markdown("""
    <style>
        .login-box {
            background-color: #1e293b; 
            padding: 30px;
            border-radius: 10px;
            box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.5);
            max-width: 400px;
            margin: auto;
        }
        h2 {
            color: #10b981 !important;
            text-align: center;
            margin-bottom: 20px;
        }
        /* Style the input fields */
        [data-testid="stTextInput"] {
            margin-bottom: 15px;
        }
    </style>
    <div class="login-box">
    <h2>🔒 Admin Login</h2>
    """, unsafe_allow_html=True)
    
    with st.form("login_form"):
        username = st.text_input("Username:", key="username_input")
        password = st.text_input("Password:", type="password", key="password_input")
        
        # Extra Extension Field (e.g., Environment Selector)
        st.selectbox(
            "Select Environment:", 
            ["Production", "Staging", "Development"],
            key="env_select"
        )
        
        st.markdown("---")
        submitted = st.form_submit_button("Log In")

    if submitted:
        if username == VALID_USERNAME and password == VALID_PASSWORD:
            # Set a session state variable upon success
            st.session_state["authenticated"] = True
            st.rerun() # Rerun the script to load the main dashboard
        else:
            st.error("Invalid username or password.")
            
    st.markdown("</div>", unsafe_allow_html=True)

# --- 2. Main Dashboard (Your Project Code) ---
def admin_dashboard():
    # Placeholder for your main dashboard logic
    # You would typically import and run the code from 'Extract Relation.py' here
    st.title("✅ Welcome to the Admin Dashboard")
    st.markdown("---")
    st.success(f"Logged in as {st.session_state['username']}.")
    
    if st.button("Log Out"):
        st.session_state["authenticated"] = False
        st.session_state["username"] = None
        st.rerun()
        
    # --- Integration Point ---
    # Call the function containing your actual dashboard code here
    # Example: run_main_dashboard() 
    # For now, we'll just show the header and a sample graph.
    st.header("Graph Data Preview")
    st.image("http://googleusercontent.com/image_collection/image_retrieval/some_id_string", caption="Sample Knowledge Graph") # 


# --- 3. Main Application Logic ---
# Initialize session state for authentication
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
    st.session_state["username"] = None

if st.session_state["authenticated"]:
    # Save the username for display on the dashboard
    st.session_state["username"] = st.session_state.get("username_input", "Admin User")
    admin_dashboard()
else:
    login_page()