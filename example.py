"""
Example usage of the reranker.
"""

from reranker import Reranker, BiEncoderReranker


def example_cross_encoder():
    """Example using cross-encoder reranker (more accurate)."""
    print("=" * 60)
    print("Cross-Encoder Reranker Example")
    print("=" * 60)
    
    # Initialize reranker
    reranker = Reranker()
    
    # Example query and documents
    query = "What is machine learning?"
    
    documents = [
        "Machine learning is a subset of artificial intelligence.",
        "Python is a programming language used for data science.",
        "Deep learning uses neural networks with multiple layers.",
        "Machine learning algorithms learn from data without explicit programming.",
        "Cooking recipes often involve following step-by-step instructions.",
        "Supervised learning uses labeled training data.",
    ]
    
    print(f"\nQuery: {query}\n")
    print("Original documents:")
    for i, doc in enumerate(documents, 1):
        print(f"{i}. {doc}")
    
    # Rerank documents
    print("\n" + "-" * 60)
    print("Reranked results (top 3):")
    print("-" * 60)
    
    reranked = reranker.rerank(query, documents, top_k=3, return_scores=True)
    
    for i, (doc, score) in enumerate(reranked, 1):
        print(f"{i}. [Score: {score:.4f}] {doc}")
    
    # Single document scoring
    print("\n" + "-" * 60)
    print("Individual document scores:")
    print("-" * 60)
    for doc in documents[:3]:
        score = reranker.score(query, doc)
        print(f"Score: {score:.4f} - {doc[:50]}...")


def example_bi_encoder():
    """Example using bi-encoder reranker (faster)."""
    print("\n" + "=" * 60)
    print("Bi-Encoder Reranker Example")
    print("=" * 60)
    
    # Initialize bi-encoder reranker
    reranker = BiEncoderReranker()
    
    query = "artificial intelligence and neural networks"
    
    documents = [
        "Neural networks are computing systems inspired by biological neural networks.",
        "Artificial intelligence enables machines to perform tasks requiring human intelligence.",
        "The weather today is sunny with a temperature of 75 degrees.",
        "Deep neural networks have revolutionized computer vision and NLP.",
        "AI systems can process natural language and understand context.",
    ]
    
    print(f"\nQuery: {query}\n")
    
    reranked = reranker.rerank(query, documents, top_k=3, return_scores=True)
    
    print("Top 3 reranked results:")
    for i, (doc, score) in enumerate(reranked, 1):
        print(f"{i}. [Score: {score:.4f}] {doc}")


def example_batch_reranking():
    """Example of batch reranking multiple queries."""
    print("\n" + "=" * 60)
    print("Batch Reranking Example")
    print("=" * 60)
    
    reranker = Reranker()
    
    queries = [
        "What is Python?",
        "How does machine learning work?",
    ]
    
    documents_list = [
        [
            "Python is a high-level programming language.",
            "Java is another programming language.",
            "Python is known for its simplicity and readability.",
            "Cooking involves preparing food.",
        ],
        [
            "Machine learning uses algorithms to learn from data.",
            "Cooking recipes help you prepare meals.",
            "ML models improve with more training data.",
            "Neural networks are used in deep learning.",
        ],
    ]
    
    results = reranker.rerank_batch(queries, documents_list, top_k=2, return_scores=True)
    
    for query, reranked_docs in zip(queries, results):
        print(f"\nQuery: {query}")
        print("Top 2 results:")
        for i, (doc, score) in enumerate(reranked_docs, 1):
            print(f"  {i}. [Score: {score:.4f}] {doc}")


if __name__ == "__main__":
    # Run examples
    example_cross_encoder()
    example_bi_encoder()
    example_batch_reranking()
    
    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
