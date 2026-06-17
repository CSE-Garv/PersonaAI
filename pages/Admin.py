import streamlit as st
import os
from src.manager import save_personality, load_personalities, delete_personality
from src.generator import auto_generate_system_prompt
from src.ingest import load_and_tag_books # Import your existing ingestion logic logic
from src.config import DATA_PATH

st.set_page_config(page_title="Admin Console", layout="wide")

if not st.session_state.get("authenticated"):
    st.warning("Please login at Home page first.")
    st.stop()

st.title("⚙️ Persona Factory")

tab1, tab2 = st.tabs(["🆕 Create New", "✏️ Edit Existing"])

with tab1:
    st.header("Forge a New Identity")
    char_name = st.text_input("Character Name (e.g., Draco Malfoy)")
    uploaded_file = st.file_uploader("Upload Source Material (PDF/TXT)", type=["pdf", "txt"])
    
    if st.button("🔮 Auto-Generate Prompt"):
        if not uploaded_file or not char_name:
            st.error("Upload a file and name the character first!")
        else:
            # 1. Read file preview
            import fitz # PyMuPDF
            doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
            text_preview = chr(12).join([page.get_text() for page in doc][:3])
            
            # 2. Generate
            with st.spinner("Analyzing psychology..."):
                generated_prompt = auto_generate_system_prompt(char_name, text_preview)
                st.session_state.gen_prompt = generated_prompt
                st.success("Prompt Generated!")

    # Prompt Editor
    final_prompt = st.text_area("System Prompt", value=st.session_state.get("gen_prompt", ""), height=300)
    
    if st.button("💾 Save Persona"):
        # Save file to disk
        save_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Save Metadata
        save_personality(char_name, uploaded_file.name, final_prompt, "Custom Character")
        
        # Trigger Ingestion (Indexing)
        with st.status("Ingesting Knowledge..."):
            # You would call your actual ingestion function here
            st.write(f"Indexing {uploaded_file.name}...")
            # load_and_tag_books() <--- UNCOMMENT THIS TO RUN INGESTION
        
        st.success(f"Character '{char_name}' created successfully!")

with tab2:
    st.header("Manage Personalities")
    personas = load_personalities()
    selected_p = st.selectbox("Select Character to Edit", list(personas.keys()))
    
    if selected_p:
        st.info(f"Source: {personas[selected_p]['book_source']}")
        new_prompt = st.text_area("Edit Prompt", personas[selected_p]['system_prompt'], height=200)
        
        col1, col2 = st.columns(2)
        if col1.button("Update"):
            save_personality(selected_p, personas[selected_p]['book_source'], new_prompt, "Updated")
            st.success("Updated!")
        if col2.button("Delete", type="primary"):
            delete_personality(selected_p)
            st.rerun()