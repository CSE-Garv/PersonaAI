# src/semantic_cache.py
"""
Semantic Cache for Chronos Bot.
Bypasses the full RAG pipeline (Cross-Encoder, SpaCy, Emotion model)
when a highly similar query has already been answered.
"""

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings


class SemanticCache:
    """
    In-memory semantic cache that stores query-response pairs
    and returns cached responses for similar queries.
    
    Cache entries are scoped by 'year' so that the same question
    asked at different timeline positions returns different results.
    """

    def __init__(self, embedding_model: HuggingFaceEmbeddings, threshold: float = 0.92):
        """
        Args:
            embedding_model: The HuggingFace embeddings model (reused from the main app).
            threshold: Cosine similarity threshold (0-1). Higher = stricter matching.
        """
        self.embedding_model = embedding_model
        self.threshold = threshold
        self.cache = []  # List of cache entry dicts

    @staticmethod
    def _cosine_similarity(vec_a, vec_b):
        """Compute cosine similarity between two vectors."""
        a = np.array(vec_a)
        b = np.array(vec_b)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def get_cached_response(self, query: str, year: int):
        """
        Check if a similar query (for the same year) exists in cache.
        
        Args:
            query: The user's input query.
            year: The selected timeline year (1-7).
            
        Returns:
            A dict with {answer, graph_html_path, context_docs} if cache hit,
            or None if no match found.
        """
        if not self.cache:
            return None

        query_embedding = self.embedding_model.embed_query(query)

        best_score = 0.0
        best_entry = None

        for entry in self.cache:
            # Must match the same year
            if entry["year"] != year:
                continue

            score = self._cosine_similarity(query_embedding, entry["query_embedding"])
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.threshold and best_entry is not None:
            return {
                "answer": best_entry["answer"],
                "graph_html_path": best_entry["graph_html_path"],
                "context_docs": best_entry["context_docs"],
                "similarity_score": best_score,
                "original_query": best_entry["query"],
            }

        return None

    def store_response(self, query: str, answer: str, graph_html_path, context_docs, year: int):
        """
        Store a new query-response pair in the cache.
        
        Args:
            query: The user's input query.
            answer: The generated answer from the RAG chain.
            graph_html_path: Path to the generated graph HTML (or None).
            context_docs: The list of retrieved context documents.
            year: The selected timeline year.
        """
        query_embedding = self.embedding_model.embed_query(query)

        self.cache.append({
            "query": query,
            "query_embedding": query_embedding,
            "answer": answer,
            "graph_html_path": graph_html_path,
            "context_docs": context_docs,
            "year": year,
        })

    def clear(self):
        """Clear the entire cache."""
        self.cache = []

    def size(self):
        """Return the number of entries in the cache."""
        return len(self.cache)
