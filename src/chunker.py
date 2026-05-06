"""
chunker.py
----------
Splits loaded documents into smaller, overlapping chunks.
This is necessary because embedding models have token limits,
and smaller chunks lead to more precise retrieval.
"""

from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_classic.schema import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100,
) -> list[Document]:
    """
    Split a list of Documents into smaller chunks.

    Args:
        documents:     List of Document objects from the loader.
        chunk_size:    Max characters per chunk (default: 500).
        chunk_overlap: Characters shared between adjacent chunks (default: 100).

    Returns:
        A new list of (smaller) Document objects with updated metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split on these separators in order (most preferred first)
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = splitter.split_documents(documents)

    # Add a chunk index to metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"Split {len(documents)} document(s) into {len(chunks)} chunks.")
    print(f"  chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    return chunks


def preview_chunks(chunks: list[Document], n: int = 3) -> None:
    """Print a preview of the first n chunks."""
    print(f"\n--- Preview of first {n} chunks ---")
    for i, chunk in enumerate(chunks[:n]):
        print(f"\n[Chunk {i}] source={chunk.metadata.get('source')} | "
              f"chars={len(chunk.page_content)}")
        print(chunk.page_content[:200])
        print("...")


if __name__ == "__main__":
    # Quick test: load docs and chunk them
    from src.loader import load_documents

    docs = load_documents("docs")
    chunks = chunk_documents(docs)
    preview_chunks(chunks)
