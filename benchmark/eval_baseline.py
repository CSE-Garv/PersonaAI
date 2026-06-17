import json
import os
import sys
import time
from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import answer_relevancy

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.config import GROQ_API_KEY
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from ragas.llms import LangchainLLMWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper

def eval_baseline():
    print("🚀 Running Baseline (No-RAG) Generation & Evaluation...")
    
    # 1. Load dataset
    dataset_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        golden_data = json.load(f)
        
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.0
    )
    
    # Simple persona prompt without context
    system_prompt = (
        "You are Harry Potter (Year {year}). "
        "Answer the user's question from your perspective."
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    chain = prompt | llm
    
    # 2. Generate Answers
    generated_data = []
    for item in golden_data:
        q = item["question"]
        year = item["year"]
        gt = item["ground_truth"]
        
        try:
            response = chain.invoke({"year": year, "input": q})
            answer = response.content
        except Exception as e:
            answer = f"ERROR: {e}"
            
        print(f"[{year}] Q: {q}")
        print(f"   A: {answer[:80]}...\n")
        
        generated_data.append({
            "question": q,
            "answer": answer,
            "contexts": [""], # Empty context for No-RAG
            "ground_truth": gt
        })
        time.sleep(1) # Respect API limits
        
    # 3. Evaluate with RAGAS
    dataset = Dataset.from_dict({
        "question": [r["question"] for r in generated_data],
        "answer": [r["answer"] for r in generated_data],
        "contexts": [r["contexts"] for r in generated_data],
        "ground_truth": [r["ground_truth"] for r in generated_data],
    })
    
    eval_llm = LangchainLLMWrapper(llm) # Reuse the same LLM
    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )
    
    print("\nEvaluating Baseline Relevancy...")
    eval_result = evaluate(
        dataset=dataset,
        metrics=[answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False
    )
    
    print("\nBASELINE METRICS:")
    if hasattr(eval_result, '_repr_dict'):
        scores_dict = eval_result._repr_dict
    else:
        import ast
        try:
            scores_dict = ast.literal_eval(str(eval_result))
        except:
            scores_dict = {"results": str(eval_result)}
            
    for k, v in scores_dict.items():
        print(f"{k}: {v:.4f}")

if __name__ == "__main__":
    eval_baseline()
