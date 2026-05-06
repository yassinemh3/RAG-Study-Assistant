"""
prompt.py
---------
Defines the prompt templates used in the RAG pipeline.

A well-crafted prompt is critical for:
  - Grounding answers in the retrieved context (preventing hallucination)
  - Getting concise, student-friendly responses
  - Handling edge cases (e.g., question not in docs)
"""

from langchain_classic.prompts import PromptTemplate, ChatPromptTemplate
from langchain_classic.schema import SystemMessage, HumanMessage


# ---------------------------------------------------------------------------
# Core RAG Prompt Template
# ---------------------------------------------------------------------------
# {context}  → filled with the retrieved chunks (from retriever.py)
# {question} → filled with the user's question

RAG_PROMPT_TEMPLATE = """You are a helpful and knowledgeable study assistant.
Your job is to answer student questions based ONLY on the provided study material.

Guidelines:
- Answer clearly and concisely using the context below.
- If the context does not contain enough information to answer, say:
  "I couldn't find information about that in your study materials."
- Do NOT make up facts or use knowledge outside the provided context.
- When helpful, use bullet points or numbered lists for clarity.
- Cite the source document name at the end of your answer.

Context from study materials:
------------------------------
{context}
------------------------------

Student Question: {question}

Answer:"""


def get_rag_prompt() -> PromptTemplate:
    """
    Returns a LangChain PromptTemplate for the RAG pipeline.

    The template expects two variables:
      - context:  The formatted retrieved chunks
      - question: The user's question

    Returns:
        A LangChain PromptTemplate instance.
    """
    return PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )


def build_prompt_text(context: str, question: str) -> str:
    """
    Manually render the prompt as a plain string.
    Useful for debugging — you can print and inspect the exact prompt sent to the LLM.

    Args:
        context:  Formatted retrieved context string.
        question: The user's question.

    Returns:
        The fully rendered prompt string.
    """
    prompt = get_rag_prompt()
    return prompt.format(context=context, question=question)


if __name__ == "__main__":
    # Preview what the prompt looks like before sending to the LLM
    sample_context = """[Source 1: ml_notes.txt]
Overfitting: The model learns training data too well and fails on new data (high variance).
Underfitting: The model is too simple to capture patterns (high bias).
Solution: Regularization, cross-validation, more data.

---

[Source 2: ml_notes.txt]
Training set: Used to fit the model.
Validation set: Used to tune hyperparameters.
Test set: Final evaluation on unseen data."""

    sample_question = "What is the difference between overfitting and underfitting?"

    rendered = build_prompt_text(sample_context, sample_question)

    print("=" * 60)
    print("RENDERED PROMPT (what gets sent to the LLM)")
    print("=" * 60)
    print(rendered)
    print("=" * 60)
    print(f"Total characters: {len(rendered)}")
