# ⚡ Chronos Bot: Advanced Persona-Driven RAG

Chronos Bot is a specialized, persona-driven Retrieval-Augmented Generation (RAG) system. It allows users to chat with "Harry Potter," constrained by a specific "Year" (book). The bot must adopt Harry's exact mentality, knowledge limitations, and emotional state from that time period to prevent **temporal hallucinations** (i.e., leaking future plot points before they happen).

## ✨ Features

- **Hybrid Ensemble Retrieval:** Combines BM25 (Sparse/Keyword) with ChromaDB (Dense/Semantic) using `all-MiniLM-L6-v2` embeddings for highly accurate lore retrieval.
- **Cross-Encoder Re-Ranking:** Uses `ms-marco-MiniLM-L-6-v2` to algorithmically score and filter the retrieved context, sending only the most relevant chunks to the LLM.
- **Semantic Caching:** An in-memory cache that evaluates query cosine similarity to instantly return answers for previously asked questions, bypassing the LLM API completely.
- **Temporal Constraint:** Prompts dynamically adjust to the selected "Year" to ensure strict adherence to the timeline.
- **Knowledge Graph UI:** Dynamically visualizes character relationships based on the text context using SpaCy and PyVis.

---

## 🚀 Setup & Installation

### 1. Clone & Install
Ensure you have Python 3.10+ installed.

```bash
git clone https://github.com/yourusername/Chronos_Bot.git
cd Chronos_Bot
pip install -r requirements.txt
```

### 2. Download NLP Models
The Knowledge Graph requires the SpaCy English core model:
```bash
python -m spacy download en_core_web_sm
```

### 3. Environment Configuration
Copy the example environment file and add your Groq API key:
```bash
cp .env.example .env
```
Open `.env` and fill in your `GROQ_API_KEY`.

### 4. Data Ingestion
Place your PDF or text files into the `data/` directory. Run the ingestion script to build the ChromaDB vector store:
```bash
python src/ingest.py
```

### 5. Run the Application
Launch the Streamlit interface:
```bash
streamlit run Home.py
```

---

## 📊 RAG Evaluation Metrics

Chronos Bot utilizes the **RAGAS** algorithmic evaluation framework to score the effectiveness of the pipeline against a curated Golden Dataset. 

Compared to a Basic RAG setup (Vanilla ChromaDB, no re-ranking), the Chronos architecture provides a massive boost:

| Metric | Basic RAG | **Chronos (Advanced RAG)** | Improvement |
| :--- | :--- | :--- | :--- |
| **Context Precision** | `0.5000` | **`0.6667`** | +33% (CrossEncoder strips irrelevant lore) |
| **Faithfulness** | `0.3190` | **`0.5839`** | +83% (Cleaner context prevents guessing) |
| **Answer Relevancy** | `0.5170` | **`0.7534`** | +45% (Highly relevant, persona-driven answers) |

For detailed analysis, read the [Evaluation Report](docs/chronos_bot_evaluation_report.md).

## 🗂 Project Structure
- `src/` - Core application logic, RAG pipelines, semantic cache, and ingestion.
- `benchmark/` - Evaluation scripts and Golden Datasets for automated RAGAS scoring.
- `docs/` - Detailed architectural whitepapers and evaluation reports.
- `data/` - Raw PDFs and text documents.
- `chroma_db/` - Persistent local vector storage.
