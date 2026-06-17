#!/usr/bin/env python3
"""
Chronos Bot — RAGAS RAG Evaluation Script
==========================================
Runs quantitative evaluation of the RAG pipeline using a curated golden dataset.
Measures: Context Precision, Context Recall, Faithfulness, Answer Relevancy.

Usage:
    cd src
    python -m benchmark.run_evaluation     (from project root, as a module)
  OR
    cd benchmark
    python run_evaluation.py               (directly)
"""

import os
import sys
import json
import time

# --- Path Setup (so we can import from src/) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SRC_DIR = os.path.join(PROJECT_ROOT, "src")

# Add src/ to sys.path so we can import app, emotion, etc.
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Add project root so relative paths like ./chroma_db work
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Change working directory to project root (for ./chroma_db, ./data paths)
os.chdir(PROJECT_ROOT)


def load_golden_dataset():
    """Load the golden dataset from benchmark/golden_dataset.json."""
    dataset_path = os.path.join(SCRIPT_DIR, "golden_dataset.json")
    if not os.path.exists(dataset_path):
        print(f"❌ Golden dataset not found at: {dataset_path}")
        sys.exit(1)

    with open(dataset_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"📚 Loaded {len(data)} evaluation questions.")
    return data


def run_rag_pipeline(dataset):
    """
    Run each question through the RAG pipeline and collect results.
    Returns a list of dicts with: question, answer, contexts, ground_truth.
    """
    from app import init_static_components, get_rag_chain
    from emotion import get_emotion, get_emotional_instruction

    print("\n🔧 Initializing RAG components...")
    embeddings, vectorstore, compressor = init_static_components()
    print("✅ Components loaded.\n")

    results = []
    total = len(dataset)

    for i, entry in enumerate(dataset, 1):
        question = entry["question"]
        ground_truth = entry["ground_truth"]
        year = entry["year"]

        print(f"  [{i}/{total}] Year {year}: \"{question[:60]}...\"")

        try:
            # Detect emotion (as the real pipeline does)
            user_emotion = get_emotion(question)
            emotion_instruction = get_emotional_instruction(user_emotion)

            # Build the RAG chain for this year
            chain = get_rag_chain(vectorstore, compressor, year, emotion_instruction)

            # Invoke with empty chat history (standalone evaluation)
            response = chain.invoke({
                "input": question,
                "chat_history": [],
                "year": year,
                "emotion_instruction": emotion_instruction,
            })

            answer = response["answer"]
            context_docs = response["context"]

            # Extract context strings for RAGAS
            contexts = [doc.page_content for doc in context_docs] if context_docs else []

            results.append({
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": ground_truth,
                "year": year,
            })

            print(f"           ✅ Answer: \"{answer[:80]}...\"")
            print(f"           📄 Retrieved {len(contexts)} context chunks.")

        except Exception as e:
            print(f"           ❌ Error: {e}")
            results.append({
                "question": question,
                "answer": f"ERROR: {e}",
                "contexts": [],
                "ground_truth": ground_truth,
                "year": year,
            })

        # Small delay to respect Groq rate limits
        time.sleep(10)

    return results


def evaluate_with_ragas(results):
    """
    Run RAGAS evaluation on the collected results.
    Returns per-question scores and aggregate metrics.
    """
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
    from langchain_groq import ChatGroq
    from langchain_community.embeddings import HuggingFaceEmbeddings

    print("\n📊 Running RAGAS evaluation...")
    print("   (This uses the Groq LLM for scoring — may take a few minutes)\n")

    # Build the HuggingFace dataset that RAGAS expects
    eval_data = {
        "question": [r["question"] for r in results],
        "answer": [r["answer"] for r in results],
        "contexts": [r["contexts"] for r in results],
        "ground_truth": [r["ground_truth"] for r in results],
    }
    dataset = Dataset.from_dict(eval_data)

    # Configure RAGAS to use the same Groq LLM
    from config import GROQ_API_KEY
    eval_llm = LangchainLLMWrapper(
        ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name="llama-3.1-8b-instant",
            temperature=0.0,  # Deterministic for evaluation
        )
    )
    eval_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    # Run evaluation
    metrics = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
    ]

    eval_result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False,
    )

    return eval_result


def print_results_table(results, eval_result):
    print("\n" + "=" * 100)
    print("  CHRONOS BOT — RAG EVALUATION RESULTS")
    print("=" * 100)

    print("\n📈 AGGREGATE SCORES:")
    print("-" * 50)
    
    # eval_result is an EvaluationResult object, we can print it directly
    # or get its internal _repr_dict if we want to format it
    if hasattr(eval_result, '_repr_dict'):
        scores_dict = eval_result._repr_dict
    else:
        # Fallback to string representation parsing if _repr_dict isn't available
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

    # Per-question details
    df = eval_result.to_pandas()
    print("\n📋 PER-QUESTION BREAKDOWN:")
    print("-" * 100)
    print(f"  {'#':>3s}  {'Year':>4s}  {'Question':<40s}  {'Faith':>6s}  {'Rel':>6s}  {'Prec':>6s}  {'Rec':>6s}")
    print("-" * 100)

    for idx, row in df.iterrows():
        q = results[idx]["question"][:38]
        year = results[idx]["year"]

        faith = row.get("faithfulness", float("nan"))
        rel = row.get("answer_relevancy", float("nan"))
        prec = row.get("context_precision", float("nan"))
        rec = row.get("context_recall", float("nan"))

        print(f"  {idx+1:3d}  {year:4d}  {q:<40s}  {faith:6.3f}  {rel:6.3f}  {prec:6.3f}  {rec:6.3f}")

    print("-" * 100)


def save_results(results, eval_result):
    """Save detailed results to benchmark/results.json."""
    output_path = os.path.join(SCRIPT_DIR, "results.json")

    # Build output data
    df = eval_result.to_pandas()
    per_question = []
    for idx, row in df.iterrows():
        per_question.append({
            "question": results[idx]["question"],
            "year": results[idx]["year"],
            "answer": results[idx]["answer"],
            "ground_truth": results[idx]["ground_truth"],
            "num_contexts": len(results[idx]["contexts"]),
            "scores": {
                "faithfulness": float(row.get("faithfulness", 0)),
                "answer_relevancy": float(row.get("answer_relevancy", 0)),
                "context_precision": float(row.get("context_precision", 0)),
                "context_recall": float(row.get("context_recall", 0)),
            }
        })

    # Global aggregate
    if hasattr(eval_result, '_repr_dict'):
        scores_dict = eval_result._repr_dict
    else:
        import ast
        try:
            scores_dict = ast.literal_eval(str(eval_result))
        except:
            scores_dict = {"results": str(eval_result)}

    output_data = {
        "aggregate_scores": scores_dict,
        "per_question": per_question
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed results saved to: {output_path}")


def main():
    print("=" * 60)
    print("  ⚡ CHRONOS BOT — RAGAS EVALUATION PIPELINE")
    print("=" * 60)

    # 1. Load golden dataset
    dataset = load_golden_dataset()

    # 2. Run each question through the RAG pipeline
    print("\n🚀 Running RAG pipeline on golden dataset...")
    results = run_rag_pipeline(dataset)

    # Filter out errored results
    valid_results = [r for r in results if not r["answer"].startswith("ERROR:")]
    if len(valid_results) < len(results):
        print(f"\n⚠️  {len(results) - len(valid_results)} questions errored out and will be excluded.")

    if not valid_results:
        print("\n❌ No valid results to evaluate. Check your pipeline and API key.")
        sys.exit(1)

    # 3. Evaluate with RAGAS
    eval_result = evaluate_with_ragas(valid_results)

    # 4. Display results
    print_results_table(valid_results, eval_result)

    # 5. Save to file
    save_results(valid_results, eval_result)

    print("\n🎉 Evaluation complete!")


if __name__ == "__main__":
    main()
