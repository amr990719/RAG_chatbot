"""
Out-of-scope detection: if retrieval can't find anything well-matched to the
question, don't waste an LLM call trying to answer it anyway.
"""

from langchain_core.documents import Document

OUT_OF_SCOPE_THRESHOLD = 0.35

OUT_OF_SCOPE_MESSAGE = (
    "I can only help with questions about FinSolve Technologies' internal data "
    "(HR, finance, marketing, engineering, and general company info) that you "
    "have access to. Could you rephrase, or ask something related to that?"
)


def is_out_of_scope(
    results: list[tuple[Document, float]], threshold: float = OUT_OF_SCOPE_THRESHOLD
) -> bool:
    if not results:
        return True
    best_score = max(score for _, score in results)
    return best_score < threshold
