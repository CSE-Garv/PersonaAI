import streamlit as st
from src.manager import load_personalities
# Import your RAG engine function
# from src.app import get_rag_chain (You will need to refactor app.py into a function)

st.set_page_config(page_title="Chat", layout="wide")

if not st.session_state.get("authenticated"):
    st.warning("Please login at Home page first.")
    st.stop()

# --- SIDEBAR: SELECTOR ---
personas = load_personalities()
if not personas:
    st.error("No personalities found! Go to Admin to create one.")
    st.stop()

selected_char = st.sidebar.selectbox("🗣️ Choose Persona", list(personas.keys()))
current_config = personas[selected_char]

st.title(f"Chat with {selected_char}")
st.caption(f"Source: {current_config['book_source']}")

# ... (Insert your existing Chat Loop Here) ...
# CRITICAL CHANGE: When calling 'get_rag_chain', pass 'current_config["system_prompt"]'
# instead of the hardcoded prompt.