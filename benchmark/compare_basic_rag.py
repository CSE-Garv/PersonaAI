import json
import os
import sys
import time
from dotenv import load_dotenv

# Add parent directory to path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import GROQ_API_KEY
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "chroma_db"))
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
    answer_relevancy,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def run_basic_rag_comparison():
    print("🚀 Running Basic RAG Generation & Evaluation...")
    
    # 1. Load Golden Dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)
        
    # 2. Setup Basic RAG Components
    # Standard embeddings and vectorstore
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    
    # Basic Retriever (No BM25, No CrossEncoder, just raw vector similarity)
    # k=2 to match the context size limits of the advanced pipeline
    retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
    
    # Standard LLM
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.0
    )
    
    # Standard System Prompt (Basic)
    system_prompt = (
        "You are Harry Potter (Year {year}). "
        "Answer the user's question from your perspective using ONLY the following context. "
        "If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    # 3. Generate Answers
    results = []
    
    for item in dataset:
        q = item["question"]
        year = item["year"]
        gt = item["ground_truth"]
        
        try:
            # Manually invoke retriever to get contexts for evaluation
            docs = retriever.invoke(q)
            context_str = format_docs(docs)
            contexts = [doc.page_content for doc in docs]
            
            # Generate answer
            response = llm.invoke(prompt.format(year=year, context=context_str, input=q))
            answer = response.content
            
        except Exception as e:
            answer = f"ERROR: {e}"
            contexts = []
            
        print(f"[{year}] Q: {q}")
        print(f"   A: {answer[:80]}...")
        print(f"   📄 Retrieved {len(contexts)} chunks.\n")
        
        results.append({
            "question": q,
            "year": year,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": gt
        })
        
        # Rate limit delay
        time.sleep(10)
        
    # 4. Ragas Evaluation
    print("\n📊 Running RAGAS Evaluation on Basic RAG...")
    
    eval_data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }
    
    hf_dataset = Dataset.from_dict(eval_data)
    
    eval_llm = LangchainLLMWrapper(
        ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.0,
        )
    )
    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]
    
    eval_result = evaluate(
        dataset=hf_dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False
    )
    
    print("\n====================================================================================================")
    print("  BASIC RAG — EVALUATION RESULTS")
    print("====================================================================================================")
    
    if hasattr(eval_result, '_repr_dict'):
        scores_dict = eval_result._repr_dict
    else:
        import ast
        try:
            scores_dict = ast.literal_eval(str(eval_result))
        except:
            scores_dict = {"results": str(eval_result)}
            
    for metric_name, score in scores_dict.items():
        if isinstance(score, (int, float)):
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"  {metric_name:25s} │ {bar} │ {score:.4f}")

    print("-" * 50)
    
if __name__ == "__main__":
    run_basic_rag_comparison()
