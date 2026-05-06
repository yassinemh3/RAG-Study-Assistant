"""
ingest.py
---------
One-time indexing script. Run this whenever you add new documents to docs/.

Pipeline:
  1. Load all PDFs/TXTs from docs/
  2. Split into overlapping chunks
  3. Embed chunks using OpenAI
  4. Save to FAISS vector store on disk

Usage:
  python ingest.py
"""

from src.loader import load_documents
from src.chunker import chunk_documents
from src.embedder import get_embedder
from src.vector_store import build_vector_store


def main():
    print("=" * 50)
    print("  Smart Study Assistant — Document Ingestion")
    print("=" * 50)

    # Step 1: Load documents
    print("\n[1/4] Loading documents from docs/...")
    docs = load_documents("docs")
    if not docs:
        print("No documents found! Add .pdf or .txt files to docs/ and try again.")
        return

    # Step 2: Chunk documents
    print("\n[2/4] Splitting into chunks...")
    chunks = chunk_documents(docs)

    # Step 3: Load embedding model
    print("\n[3/4] Loading embedding model...")
    embedder = get_embedder()

    # Step 4: Build and save vector store
    print("\n[4/4] Embedding chunks and saving vector store...")
    build_vector_store(chunks, embedder)

    print("\n" + "=" * 50)
    print("  Ingestion complete!")
    print(f"  {len(docs)} file(s) | {len(chunks)} chunks indexed")
    print("  Run: streamlit run app.py")
    print("=" * 50)


if __name__ == "__main__":
    main()
