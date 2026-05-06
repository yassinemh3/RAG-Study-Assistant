"""
embedder.py
-----------
Creates an embedding model instance.
The embedder converts text into numerical vectors (embeddings)
that capture semantic meaning. Similar texts → similar vectors.

Model: Google Gemini text-embedding-004
- 768 dimensions
- Free tier available via Google AI Studio
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Load API key from .env file
load_dotenv()


def get_embedder() -> GoogleGenerativeAIEmbeddings:
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key or api_key == "your_google_api_key_here":
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Please add your key to the .env file:\n"
            "  GOOGLE_API_KEY=AIza..."
        )

    embedder = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",  # Free Gemini embedding model, 768 dimensions
        google_api_key=api_key,
    )

    return embedder


if __name__ == "__main__":
    # Quick test: embed a single sentence and print vector info
    print("Testing embedding model...")

    embedder = get_embedder()
    test_sentence = "What is supervised learning?"

    vector = embedder.embed_query(test_sentence)

    print(f"Input : '{test_sentence}'")
    print(f"Vector dimensions : {len(vector)}")
    print(f"First 5 values    : {[round(v, 4) for v in vector[:5]]}")
    print("Embedding model is working correctly!")
