import streamlit as st
from src.config import ADMIN_USER, ADMIN_PASS

st.set_page_config(page_title="Chronos Login", page_icon="🔐")

st.title("🔐 Chronos Gatekeeper")

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if username == ADMIN_USER and password == ADMIN_PASS:
            st.session_state.authenticated = True
            st.success("Access Granted. Select a page from the sidebar.")
            st.rerun()
        else:
            st.error("Invalid Credentials")
else:
    st.success(f"Welcome, {ADMIN_USER}!")
    st.info("👈 Navigate to 'Chat' or 'Admin' using the sidebar.")