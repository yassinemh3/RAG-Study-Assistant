"""
retriever.py
------------
Wraps the FAISS vector store with retrieval logic.

Given a user's question, it finds the top-k most semantically
similar chunks from the indexed documents.
"""

from langchain_community.vectorstores import FAISS
from langchain_classic.schema import Document


def get_retriever(vector_store: FAISS, k: int = 4):
    """
    Create a LangChain retriever from the FAISS vector store.

    Using LangChain's built-in retriever interface means we can
    swap FAISS for any other vector DB later with zero code changes.

    Args:
        vector_store: A loaded FAISS vector store.
        k:            Number of chunks to retrieve per query (default: 4).

    Returns:
        A LangChain BaseRetriever instance.
    """
    retriever = vector_store.as_retriever(
        search_type="similarity",   # plain cosine/dot-product similarity
        search_kwargs={"k": k},
    )
    return retriever


def retrieve(query: str, vector_store: FAISS, k: int = 4) -> list[Document]:
    """
    Perform a direct similarity search and return the top-k chunks.

    Args:
        query:        The user's question as a plain string.
        vector_store: A loaded FAISS vector store.
        k:            Number of results to return.

    Returns:
        List of the top-k most relevant Document chunks.
    """
    results = vector_store.similarity_search(query, k=k)
    return results


def format_context(docs: list[Document]) -> str:
    """
    Merge retrieved chunks into a single context string for the LLM prompt.
    Each chunk is separated and labelled with its source file.

    Args:
        docs: List of retrieved Document chunks.

    Returns:
        A formatted multi-chunk context string.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content}")

    return "\n\n---\n\n".join(parts)


if __name__ == "__main__":
    # Test retrieval (requires vector store to be built first via ingest.py)
    from src.embedder import get_embedder
    from src.vector_store import load_vector_store

    embedder = get_embedder()
    vs = load_vector_store(embedder)

    test_queries = [
        "What is overfitting?",
        "How does gradient descent work?",
        "What are evaluation metrics for ML models?",
    ]

    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 50)
        results = retrieve(query, vs, k=2)
        for i, doc in enumerate(results, 1):
            print(f"  [{i}] {doc.metadata.get('source')} | {doc.page_content[:120]}...")
