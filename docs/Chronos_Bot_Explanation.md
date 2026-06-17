# Chronos Bot: In-Depth Project Architecture & Explanation

**Chronos Bot** (also referred to as "Chronos Gatekeeper") is an advanced, persona-driven chatbot framework that relies heavily on a cutting-edge Retrieval-Augmented Generation (RAG) architecture. Built primarily as a Streamlit web application, it leverages Llama models on the Groq inference engine to generate highly context-aware, emotionally intelligent, and logically sound responses, specifically tuned to simulate literary or fictional personas (with a recurring theme built around the Harry Potter universe).

Below is an in-depth breakdown of the project's features, architecture, and core components.

---

## 1. High-Level Architecture Overview

At its core, Chronos Bot operates by taking source materials (PDFs, TXTs), intelligently extracting and chunking their contents, and loading them into a Chroma vector database. It then uses a robust, multi-stage retrieval process and emotion-recognition pipeline to assemble context before finally prompting a specific "Persona" via an LLM.

### Tech Stack:
- **Frontend / UI:** Streamlit (`Home.py`, `pages/Chat.py`, `pages/Admin.py`)
- **LLM & Inference:** Groq API leveraging open-source models like `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`.
- **RAG & Orchestration:** LangChain (and LangGraph/experimental components).
- **Vector Database:** ChromaDB (`Chroma`).
- **Embeddings & Reranking:** HuggingFace `sentence-transformers`, `CrossEncoder`.
- **Emotion Detection:** HuggingFace `transformers` (`j-hartmann/emotion-english-distilroberta-base`).
- **Entity Extraction & Graphing:** SpaCy (`en_core_web_sm`), NetworkX, and PyVis.
- **OCR:** RapidOCR for fallback text extraction.

---

## 2. Core Modules & Functionality

### A. The User Interface (Streamlit)
The application acts as a multi-page Streamlit portal controlled by a simple session-state authentication layer:
- **`Home.py`:** The entry point and login gatekeeper for the app.
- **`pages/Admin.py` ("Persona Factory"):** A powerful admin dashboard that lets users upload source material (books/documents) and **automatically generate** a system prompt defining a character's persona, psychological profile, and logical boundary rules using an LLM (`src/generator.py`). These personas are saved to local JSON configuration via `src/manager.py`.
- **`pages/Chat.py`:** The primary interface where users select a loaded persona and interact with them. 

### B. Knowledge Ingestion Pipeline (`src/ingest.py`)
This module handles taking unstructured data into the RAG vector store.
1. **Intelligent Extraction & OCR:** Attempts to extract text using standard PDF loaders. If the document is scanned or unreadable, it gracefully falls back to `RapidOCR` to read text page-by-page.
2. **Text Cleaning:** Fixes common PDF artifacts like broken hyphenation, erratic newlines, and unexpected spaces.
3. **Semantic Chunking:** Crucially, instead of simple rigid-size chunks, it uses LangChain's `SemanticChunker` to break the text at meaningful, logical breakpoints (based on the 95th percentile of semantic shifts in the subject matter), providing vastly superior context coherence.
4. **Metadata Indexing:** Embeds the chunk with a `book_index` to facilitate temporal context boundaries later.

### C. The Advanced RAG Engine (`src/app.py`)
The logic engine of Chronos Bot utilizes a sophisticated retrieval stack to ensure highly factual and context-driven logic.
1. **Hybrid Retrieval (Ensemble):** It queries both:
   - **Dense Retrieval (ChromaDB):** For conceptual similarity using HuggingFace `all-MiniLM-L6-v2` embeddings.
   - **Sparse Retrieval (BM25):** For exact keyword matching.
2. **Temporal / Spoiler Guard:** The app enforces a "Timeline" (via user slider for Year 1 to 7). It applies metadata filtering (`"book_index": {"$lte": selected_year}`) on the vector DB to actively prevent the bot from hallucinating or accessing knowledge from "future" events.
3. **Cross-Encoder Reranking:** The retrieved documents are fed through a `ContextualCompressionRetriever` with an `ms-marco` cross-encoder to strictly re-rank the top 5 most relevant documents to feed into the final context window.
4. **History-Aware Question Reformulation:** Takes the chat history and transforms ambiguous user queries (e.g., "Are you sure?") into standalone queries to ensure continuous subject tracking.
5. **Strict Logic Prompts:** The final system prompt forces the LLM to abide by explicit "Family vs. Friends" logic definitions, entity disambiguation rules, and specific instructions to stay in character (e.g., dismissing Muggle technology).

### D. Dynamic Emotion Recognition (`src/emotion.py`)
Before sending the prompt to the LLM, Chronos Bot analyzes the user's input with a local DistilRoBERTa emotion classification model. 
- It detects 7 emotions: `joy`, `sadness`, `anger`, `fear`, `surprise`, `disgust`, or `neutral`.
- It maps the detected emotion to an **Emotional Instruction** injected directly into the LLM persona prompt (e.g., if the user is sad, the bot is prompted to "Drop the sass. Be gentle, empathetic, and comforting," creating a dynamic, highly responsive conversational partner).

### E. Live Knowledge Graphs (`src/graph_utils.py`)
As a "wow" factor and transparency feature, every time a query pulls context documents, the framework generates a live graph representation of that text:
1. **Entity Extraction (SpaCy):** Extracts people, organizations, and places from the retrieved chunks.
2. **Graph Construction (NetworkX):** Connects entities found in the same sentences to build an ad-hoc relationship graph.
3. **Visualization (PyVis):** Generates an interactive HTML physics-based visualization (`graph.html`) of the characters active in the current context, rendered directly into the Streamlit UI.

---

## 3. Summary of Workflow

1. **Setup:** Admin creates a persona (e.g., "Harry Potter") by uploading a defining book. `src/ingest.py` semantically chunks the book into ChromaDB. `src/generator.py` architects a system prompt for the persona.
2. **User Input:** A user chats via the Streamlit interface.
3. **Preprocessing:** `src/emotion.py` evaluates the emotional subtext of the user's message.
4. **Retrieval:** Historical RAG runs a hybrid vector + lexical search, filtered by the "current year", then structurally re-ranks the findings.
5. **Contextualization:** SpaCy creates an interactive graph of the retrieved context.
6. **Generation:** Groq LLM binds the persona rules, the emotional instructions, the exact text context, and outputs the response.

Chronos Bot represents a sophisticated, bleeding-edge approach to persona chatbots, eliminating standard hallucinations by using aggressive filtering, hybrid RAG, dynamic emotive context, and cross-encoder refinement.
