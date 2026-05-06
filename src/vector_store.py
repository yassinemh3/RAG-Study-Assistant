"""
vector_store.py
---------------
Handles creating, saving, and loading the FAISS vector store.

The vector store maps each chunk's embedding to its text content,
enabling fast semantic similarity search at query time.
"""

import os
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.schema import Document

VECTOR_STORE_PATH = "vector_store"


def build_vector_store(
    chunks: list[Document],
    embedder: GoogleGenerativeAIEmbeddings,
    save_path: str = VECTOR_STORE_PATH,
) -> FAISS:
    """
    Embed all chunks and store them in a FAISS index.
    Saves the index to disk so it can be reloaded later.

    Args:
        chunks:    List of text chunks from chunker.py
        embedder:  Embedding model from embedder.py
        save_path: Directory to save the FAISS index

    Returns:
        The in-memory FAISS vector store.
    """
    print(f"Building vector store from {len(chunks)} chunks...")
    print("  (This calls the OpenAI Embeddings API — may take a few seconds)")

    # This embeds all chunks and builds the FAISS index in one call
    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedder,
    )

    # Persist to disk so we don't re-embed every time
    os.makedirs(save_path, exist_ok=True)
    vector_store.save_local(save_path)
    print(f"Vector store saved to '{save_path}/'")

    return vector_store


def load_vector_store(
    embedder: GoogleGenerativeAIEmbeddings,
    load_path: str = VECTOR_STORE_PATH,
) -> FAISS:
    """
    Load a previously built FAISS index from disk.

    Args:
        embedder:  Embedding model (needed to embed new queries)
        load_path: Directory where the FAISS index was saved

    Returns:
        The loaded FAISS vector store.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"No vector store found at '{load_path}'. "
            "Run 'python ingest.py' first to build it."
        )

    print(f"Loading vector store from '{load_path}/'...")
    vector_store = FAISS.load_local(
        load_path,
        embedder,
        allow_dangerous_deserialization=True,  # Safe: we created this file ourselves
    )
    print("Vector store loaded successfully.")
    return vector_store


if __name__ == "__main__":
    # Full pipeline test: load → chunk → embed → store → reload
    from src.loader import load_documents
    from src.chunker import chunk_documents
    from src.embedder import get_embedder

    print("=== Vector Store Build Test ===\n")

    docs = load_documents("docs")
    chunks = chunk_documents(docs)
    embedder = get_embedder()

    # Build and save
    vs = build_vector_store(chunks, embedder)

    # Reload from disk and do a quick search
    vs2 = load_vector_store(embedder)
    results = vs2.similarity_search("What is gradient descent?", k=2)

    print("\n--- Top 2 results for: 'What is gradient descent?' ---")
    for i, doc in enumerate(results):
        print(f"\n[Result {i+1}] source={doc.metadata.get('source')}")
        print(doc.page_content[:200])
