"""
src/generator.py
----------------
Generates answers using retrieved context chunks from the vector store.
Uses Google Gemini as the LLM and reuses the prompt template from prompt.py.

Requires GOOGLE_API_KEY in your .env file.
Install: pip install google-generativeai
"""

import os
from dotenv import load_dotenv
import google.generativeai as genai

from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_classic.schema import Document

from src.prompt import build_prompt_text

load_dotenv()


# ---------------------------------------------------------------------------
# Internal: configure and call Gemini
# ---------------------------------------------------------------------------

def _configure_gemini() -> genai.GenerativeModel:
    """Configure the Gemini client and return a model instance."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key or api_key == "your_google_api_key_here":
        raise ValueError(
            "GOOGLE_API_KEY is not set. "
            "Please add your key to the .env file:\n"
            "  GOOGLE_API_KEY=AIza..."
        )
    genai.configure(api_key=api_key)
    return genai.GenerativeModel("gemini-2.5-flash-lite")


def _format_context(docs: list[Document]) -> str:
    """
    Format retrieved Document chunks into a single context string
    that matches the placeholder expected by prompt.py's RAG_PROMPT_TEMPLATE.
    """
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[Source {i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Public function used by app.py and ingest.py
# ---------------------------------------------------------------------------

def answer_question(
    question: str,
    vector_store: FAISS,
    k: int = 4,
    return_sources: bool = False,
) -> dict:
    """
    Retrieve relevant chunks from the vector store and generate an answer
    using Google Gemini.

    Args:
        question:      The student's question string.
        vector_store:  A loaded FAISS vector store (from vector_store.py).
        k:             Number of chunks to retrieve (default: 4).
        return_sources: If True, include source Documents in the result dict.

    Returns:
        dict with keys:
            "answer"  — the generated answer string
            "sources" — list of Document objects (only if return_sources=True)
    """

    # 1. Retrieve top-k relevant chunks via semantic search
    docs = vector_store.similarity_search(question, k=k)

    if not docs:
        return {
            "answer": (
                "I couldn't find any relevant information in your study materials "
                "to answer that question."
            ),
            "sources": [],
        }

    # 2. Format context using the same structure as prompt.py preview
    context = _format_context(docs)

    # 3. Build the full prompt via prompt.py (keeps prompts in one place)
    prompt = build_prompt_text(context=context, question=question)

    # 4. Call Gemini
    model = _configure_gemini()
    response = model.generate_content(prompt)
    answer = response.text

    # 5. Return result
    result: dict = {"answer": answer}
    if return_sources:
        result["sources"] = docs

    return result