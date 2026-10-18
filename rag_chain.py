"""
RBAC-filtered retrieval + Groq-backed generation.

retrieve_with_scores() needs no Groq key (local embeddings only) and is used
by both the scope guardrail and generate_answer().
generate_answer() needs GROQ_API_KEY set (in .env) since it calls Groq Cloud.
"""

import os

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from rbac import get_allowed_departments

load_dotenv()

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "finsolve_docs"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
RETRIEVAL_K = 10

_embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},
)

_vectorstore = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=_embeddings,
    persist_directory=PERSIST_DIR,
)

_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are FinSolve Technologies' internal assistant. Answer the "
            "question using ONLY the context below. If the context doesn't "
            "contain the answer, say you don't have that information -- do "
            "not use outside knowledge. Be concise and cite figures exactly "
            "as given.\n\nContext:\n{context}",
        ),
        ("human", "{question}"),
    ]
)

_llm = None  # built lazily so a missing GROQ_API_KEY doesn't break retrieval-only use


def _get_llm() -> ChatGroq:
    global _llm
    if _llm is None:
        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to a .env file (see .env.example)."
            )
        _llm = ChatGroq(model=GROQ_MODEL, temperature=0)
    return _llm


def retrieve_with_scores(question: str, role: str, k: int = RETRIEVAL_K) -> list[tuple[Document, float]]:
    """RBAC-filtered similarity search. Higher score = more relevant (cosine, 0-1)."""
    allowed_departments = get_allowed_departments(role)
    return _vectorstore.similarity_search_with_relevance_scores(
        question,
        k=k,
        filter={"department": {"$in": allowed_departments}},
    )


def generate_answer(question: str, docs: list[Document]) -> str:
    """Generate a grounded answer from already-retrieved (RBAC-filtered) docs."""
    if not docs:
        return "I don't have information about that in the documents I can access."

    context = "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source')}]\n{doc.page_content}" for doc in docs
    )
    chain = _prompt | _get_llm() | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
