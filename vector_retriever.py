import logging
from typing import Any, Dict, List

from chroma import ChromaManager

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class VectorRetriever:

    def __init__(self, chroma_manager: ChromaManager) -> None:
        if chroma_manager is None:
            raise ValueError("chroma_manager must be a valid ChromaManager instance.")

        self.chroma_manager = chroma_manager

    def retrieve(self, question: str, k: int = 5) -> List[Dict[str, Any]]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")

        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        try:
            results = self.chroma_manager.similarity_search(question, k=k)
        except Exception as error:
            raise RuntimeError(f"Vector similarity search failed: {error}") from error

        if not results:
            logger.info("No vector search results found for question: %s", question)
            return []

        retrieved_documents = [
            {"content": document.page_content, "metadata": document.metadata}
            for document in results
        ]

        logger.info(
            "Retrieved %d chunk(s) for question: %s", len(retrieved_documents), question
        )

        return retrieved_documents