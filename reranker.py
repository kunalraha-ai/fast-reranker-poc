"""
Reranker Module
A flexible reranker implementation for reordering search results based on relevance.
"""

from typing import List, Tuple, Optional, Union
import numpy as np
from sentence_transformers import CrossEncoder
import torch


class Reranker:
    """
    A reranker that uses cross-encoder models to rerank search results.
    
    Cross-encoders process query-document pairs together, providing
    more accurate relevance scores than bi-encoders.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
        max_length: int = 512
    ):
        """
        Initialize the reranker.
        
        Args:
            model_name: Name of the cross-encoder model to use.
                       Default is a good general-purpose reranker.
            device: Device to run the model on ('cuda', 'cpu', or None for auto).
            max_length: Maximum sequence length for the model.
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        print(f"Loading reranker model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=self.device
        )
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = False
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Rerank documents based on their relevance to the query.
        
        Args:
            query: The search query.
            documents: List of documents to rerank.
            top_k: Number of top results to return. If None, returns all.
            return_scores: If True, returns tuples of (document, score).
                          If False, returns only documents.
        
        Returns:
            Reranked list of documents (or document-score tuples).
        """
        if not documents:
            return []
        
        # Create query-document pairs
        pairs = [[query, doc] for doc in documents]
        
        # Get relevance scores
        scores = self.model.predict(pairs)
        
        # Sort by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k if specified
        if top_k is not None:
            scored_docs = scored_docs[:top_k]
        
        # Return format based on return_scores flag
        if return_scores:
            return scored_docs
        else:
            return [doc for doc, _ in scored_docs]
    
    def rerank_batch(
        self,
        queries: List[str],
        documents_list: List[List[str]],
        top_k: Optional[int] = None,
        return_scores: bool = False,
        batch_size: int = 32
    ) -> List[Union[List[str], List[Tuple[str, float]]]]:
        """
        Rerank multiple query-document sets in batch.
        
        Args:
            queries: List of queries.
            documents_list: List of document lists, one per query.
            top_k: Number of top results to return per query.
            return_scores: If True, returns tuples of (document, score).
            batch_size: Batch size for processing.
        
        Returns:
            List of reranked results, one per query.
        """
        results = []
        for query, documents in zip(queries, documents_list):
            result = self.rerank(query, documents, top_k, return_scores)
            results.append(result)
        return results
    
    def score(
        self,
        query: str,
        document: str
    ) -> float:
        """
        Get relevance score for a single query-document pair.
        
        Args:
            query: The search query.
            document: The document to score.
        
        Returns:
            Relevance score (higher is more relevant).
        """
        score = self.model.predict([[query, document]])
        return float(score[0])


class BiEncoderReranker:
    """
    Alternative reranker using bi-encoder architecture.
    Faster but potentially less accurate than cross-encoder.
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None
    ):
        """
        Initialize the bi-encoder reranker.
        
        Args:
            model_name: Name of the sentence transformer model.
            device: Device to run the model on.
        """
        from sentence_transformers import SentenceTransformer
        
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Loading bi-encoder model: {model_name}")
        print(f"Using device: {self.device}")
        
        self.model = SentenceTransformer(model_name, device=self.device)
    
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_k: Optional[int] = None,
        return_scores: bool = False
    ) -> Union[List[str], List[Tuple[str, float]]]:
        """
        Rerank documents using cosine similarity.
        
        Args:
            query: The search query.
            documents: List of documents to rerank.
            top_k: Number of top results to return.
            return_scores: If True, returns tuples of (document, score).
        
        Returns:
            Reranked list of documents (or document-score tuples).
        """
        if not documents:
            return []
        
        # Encode query and documents
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        doc_embeddings = self.model.encode(documents, convert_to_numpy=True)
        
        # Compute cosine similarity
        scores = np.dot(doc_embeddings, query_embedding) / (
            np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # Sort by score (descending)
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Apply top_k if specified
        if top_k is not None:
            scored_docs = scored_docs[:top_k]
        
        # Return format based on return_scores flag
        if return_scores:
            return scored_docs
        else:
            return [doc for doc, _ in scored_docs]
# --- RUN THIS TO TEST ---
if __name__ == "__main__":
    print("Initializing Reranker...")
    
    # 1. Start the Engine
    ranker = Reranker()
    
    # 2. Define the Test Data
    user_query = "How to save money on OpenAI API costs?"
    candidates = [
        "The weather in Pune is nice today.",
        "You can use a Cross-Encoder to filter documents before sending to LLM.",
        "Arlecchino is a Pyro DPS character in Genshin Impact.",
        "Reranking reduces token usage by sending fewer docs to GPT-4."
    ]
    
    # 3. Run the Ranker
    print(f"\nQuery: {user_query}")
    print("Ranking documents...\n")
    
    # FIX: We use .rerank() because that is what your class uses
    # We set return_scores=True so we get the numbers
    results = ranker.rerank(user_query, candidates, return_scores=True)
    
    # 4. Print Winners
    for item in results:
        # Handling different return formats just in case
        if isinstance(item, (list, tuple)) and len(item) == 2:
            text, score = item
            print(f"Score: {score:.4f} | {text}")
        else:
            print(f"Result: {item}")            