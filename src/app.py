import os
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser

from config import GROQ_API_KEY, DB_PATH


# --- CORE FUNCTIONS (importable by eval script) ---

def init_static_components(db_path=None):
    """
    Initialize the embedding model, vector store, and cross-encoder compressor.
    
    Args:
        db_path (str, optional): The path to the ChromaDB directory. Defaults to DB_PATH.
        
    Returns:
        tuple: (embeddings, vectorstore, compressor)
    """
    if db_path is None:
        db_path = DB_PATH
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=db_path, embedding_function=embeddings)

    # Judge Model for cross-encoder reranking
    model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=model, top_n=2)

    return embeddings, vectorstore, compressor


def get_rag_chain(vectorstore, compressor, selected_year, emotion_instruction):
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.3 # Balanced for creativity + facts
    )
    )
    
    # 1. RETRIEVAL PIPELINE
    chroma_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 20, "filter": {"book_index": {"$lte": selected_year}}}
    )

    try:
        allowed_docs = vectorstore.get(where={"book_index": {"$lte": selected_year}})
        if len(allowed_docs['documents']) > 0:
            filtered_docs = [Document(page_content=t, metadata=m) for t, m in zip(allowed_docs['documents'], allowed_docs['metadatas'])]
            bm25_retriever = BM25Retriever.from_documents(filtered_docs)
            bm25_retriever.k = 10
            ensemble_retriever = EnsembleRetriever(
                retrievers=[chroma_retriever, bm25_retriever], weights=[0.5, 0.5]
            )
        else:
            ensemble_retriever = chroma_retriever
    except:
        ensemble_retriever = chroma_retriever

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever
    )

    # 2. HISTORY-AWARE REFORMULATION
    # Generic logic to handle "Are you sure?" without naming Snape
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "formulate a standalone question. \n\n"
        "### LOGIC RULES ###\n"
        "1. **Ambiguity Resolution:** If the user asks 'Are you sure?', identify the *Fact* asserted in the bot's last message and create a verification question (e.g., 'Verify if [Fact] is supported by text').\n"
        "2. **Subject Continuity:** Ensure 'he/she/they' refers to the *primary subject* of the conversation flow."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, compression_retriever, contextualize_q_prompt
    )

    # 3. SYSTEM PROMPT & CHAIN SETUP

    system_prompt = (
        "You are Harry Potter (Year {year}). "
        "Answer using the Context below. "
        "\n\n"
        "### LOGIC PROTOCOL: FAMILY VS. FRIENDS ###\n"
        "The user often confuses 'Close Friends' with 'Family'. You must distinguish them using these strict definitions:\n"
        "1.  **Strict Family Definition:** classify a character as 'Family' ONLY if the text explicitly states a **Blood Relation** (Mother, Father, Uncle, Cousin) or **Legal Guardianship** (Godfather, Adoptive Parents).\n"
        "2.  **The 'Like a Brother' Trap:** If the text says someone is 'like a brother' or 'part of the family' (metaphorically), categorize them as **FRIENDS**, not relatives.\n"
        "3.  **Ambiguity Check:** If the text mentions a character (e.g., 'Petunia') but does not explicitly state their relationship to you, do NOT guess. Say 'I know Petunia, but the text doesn't say how we are related.'\n"
        "\n"
        "### ENTITY DISAMBIGUATION PROTOCOL ###\n"
        "1. **Scan:** Identify all characters mentioned in the Context.\n"
        "2. **Map:** Explicitly check which role belongs to which name (e.g., Who is the 'friend'? Who is the 'enemy'?). Do not mix them up.\n"
        "3. **Verify:** If the user asks 'Are you sure?', re-read the text strictly. If the text says 'X is Y's friend' and you said 'Z is Y's friend', CORRECTION is required.\n"
        "4. **Truth:** If the context is empty or unclear, say 'I don't know'. Do not guess relationships.\n"
        "\n"
        "### TEMPORAL INTEGRITY (SPOILER GUARD) ###\n"
        "- **Future Knowledge:** You do NOT know what happens in future years. (e.g., In Year 1, you do not know about Sirius Black or Horcruxes).\n"
        "- **If asked about the future:** Speculate wildly or say 'I haven't got a clue, mate. Trelawney might know.'\n"
        "\n"
        "### WORLD GROUNDING ###\n"
        "- **Muggle Tech:** If asked about iPhones, Internet, or AI, act confused. You are a wizard in the 1990s. Call them 'Muggle contraptions'.\n"
        "- **Spells:** If the user casts a spell (e.g., 'Expelliarmus'), REACT to it. Do not define it. (e.g., 'Whoa! Nearly took my glasses off!').\n"
        "\n"
        "### PERSONA ###\n"
        "Use British teen slang if appropriate, but never sacrifice accuracy for tone.\n"
        "Emotion to apply: {emotion_instruction}\n"
        "\n"
        "Context: {context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    return rag_chain


# --- STREAMLIT UI (only runs when executed directly, not when imported) ---

def main():
    """Main Streamlit UI entry point."""
    import streamlit as st
    import streamlit.components.v1 as components
    from emotion import get_emotion, get_emotional_instruction
    from graph_utils import build_graph_from_text, save_graph_html
    from semantic_cache import SemanticCache
    from config import CACHE_SIMILARITY_THRESHOLD

    st.set_page_config(page_title="Chronos: Logic-Driven RAG", page_icon="⚡")

    # --- Cached resource loader (Streamlit-specific) ---
    @st.cache_resource
    def get_static_components():
        return init_static_components()

    # --- Initialize Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "semantic_cache" not in st.session_state:
        # Initialize cache with the same embedding model used everywhere
        embeddings, _, _ = get_static_components()
        st.session_state.semantic_cache = SemanticCache(
            embedding_model=embeddings,
            threshold=CACHE_SIMILARITY_THRESHOLD
        )

    cache = st.session_state.semantic_cache

    # --- UI LAYOUT ---
    st.title("⚡ Chronos: Logic-Driven Architecture")
    st.caption("No Hardcoding • Entity Disambiguation • Hybrid RAG • Semantic Cache")

    with st.sidebar:
        st.header("⏳ Timeline")
        selected_year = st.slider("Year", 1, 7, 1)

        # Cache stats
        st.divider()
        st.metric("🧠 Cache Entries", cache.size())

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear Memory"):
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Clear Cache"):
                cache.clear()
                st.rerun()

    # --- Display Chat History ---
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # --- Chat Input ---
    if prompt := st.chat_input("Ask Harry..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()

            try:
                # --- SEMANTIC CACHE CHECK ---
                cached = cache.get_cached_response(prompt, selected_year)

                if cached is not None:
                    # ⚡ CACHE HIT — bypass the entire heavy pipeline
                    answer = cached["answer"]
                    source_docs = cached["context_docs"]
                    graph_html_path = cached["graph_html_path"]

                    message_placeholder.markdown(answer)
                    st.toast(
                        f"⚡ **Cached!** Similarity: {cached['similarity_score']:.1%} "
                        f"to \"{cached['original_query'][:50]}...\"",
                        icon="⚡"
                    )
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                else:
                    # --- CACHE MISS — run the full pipeline ---
                    message_placeholder.markdown("⚡ *Thinking...*")

                    embeddings, vectorstore, compressor = get_static_components()
                    user_emotion = get_emotion(prompt)
                    emotion_instruction = get_emotional_instruction(user_emotion)

                    chain = get_rag_chain(vectorstore, compressor, selected_year, emotion_instruction)

                    # Send limited history to save tokens
                    recent_history = st.session_state.messages[-4:]

                    response = chain.invoke({
                        "input": prompt,
                        "chat_history": recent_history,
                        "year": selected_year,
                        "emotion_instruction": emotion_instruction
                    })

                    answer = response['answer']
                    source_docs = response['context']

                    # --- GRAPH GENERATION START ---
                    graph_html_path = None
                    if source_docs:
                        full_context_text = " ".join([d.page_content for d in source_docs])

                        # 🔍 DEBUG: Show us what entities Spacy is finding
                        import spacy
                        nlp = spacy.load("en_core_web_sm")
                        doc = nlp(full_context_text)
                        found_entities = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG"]]

                        # Check if we found anything
                        if not found_entities:
                             st.warning("⚠️ Graph Debug: No 'People' found in text. Spacy might need an update.")
                        elif len(set(found_entities)) < 2:
                             st.info(f"⚠️ Graph Debug: Found only 1 entity ({found_entities[0]}). Need 2 to connect.")
                        else:
                             # Proceed to build graph
                             G = build_graph_from_text(full_context_text)
                             graph_html_path = save_graph_html(G)
                    # --- GRAPH GENERATION END ---

                    # Display Answer
                    message_placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

                    # --- STORE IN CACHE ---
                    cache.store_response(
                        query=prompt,
                        answer=answer,
                        graph_html_path=graph_html_path,
                        context_docs=source_docs,
                        year=selected_year
                    )

                # --- Display Graph (for both cached and fresh responses) ---
                if graph_html_path:
                    with st.expander("🕸️ Live Entity Graph (Interactive)", expanded=True):
                        # Read and render the HTML file
                        with open(graph_html_path, 'r', encoding='utf-8') as f:
                            html_string = f.read()
                        components.html(html_string, height=360, scrolling=False)

                # Display Sources
                if source_docs:
                    with st.expander(f"📚 Context Analyzed ({len(source_docs)} Chunks)"):
                        for i, doc in enumerate(source_docs):
                            st.caption(f"**Source {i+1}**: {doc.page_content[:200]}...")
                else:
                    with st.expander("⚠️ No Context"):
                        st.write("Bot answered based on internal logic (or refused to answer).")

            except Exception as e:
                message_placeholder.error(f"Error: {e}")


if __name__ == "__main__":
    main()