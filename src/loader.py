"""
loader.py
---------
Loads PDF and plain-text documents from a folder.
Returns a list of LangChain Document objects.
"""

import os
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_classic.schema import Document


SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


def load_documents(docs_dir: str = "docs") -> list[Document]:
    """
    Scan the given directory and load all supported files.

    Args:
        docs_dir: Path to the folder containing your study materials.

    Returns:
        A list of LangChain Document objects.
    """
    docs_path = Path(docs_dir)

    if not docs_path.exists():
        raise FileNotFoundError(f"Directory '{docs_dir}' not found. Please create it and add your files.")

    all_documents = []
    found_files = []

    # Walk through every file in the docs folder
    for file_path in sorted(docs_path.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            print(f"  [SKIP] Unsupported file type: {file_path.name}")
            continue

        found_files.append(file_path.name)
        print(f"  [LOAD] Loading: {file_path.name}")

        try:
            if file_path.suffix.lower() == ".pdf":
                # PyPDFLoader splits by page automatically
                loader = PyPDFLoader(str(file_path))
            else:
                # TextLoader reads the whole file as one document
                loader = TextLoader(str(file_path), encoding="utf-8")

            docs = loader.load()

            # Enrich metadata with the source filename
            for doc in docs:
                doc.metadata["source"] = file_path.name

            all_documents.extend(docs)

        except Exception as e:
            print(f"  [ERROR] Failed to load {file_path.name}: {e}")

    # Summary
    print(f"\nLoaded {len(all_documents)} document page(s) from {len(found_files)} file(s).")
    return all_documents


if __name__ == "__main__":
    # Quick test: run this file directly to verify loading works
    docs = load_documents("docs")
    print("\n--- Preview of first document ---")
    if docs:
        print(f"Source : {docs[0].metadata.get('source', 'unknown')}")
        print(f"Content: {docs[0].page_content[:300]}...")
    else:
        print("No documents found. Add .pdf or .txt files to the 'docs/' folder.")
