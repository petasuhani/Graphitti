import logging
from typing import Any, Dict, List

from graph_retriever import GraphRetriever
from vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HybridRetriever:
    def __init__(self, graph_retriever: GraphRetriever, vector_retriever: VectorRetriever) -> None:
        if graph_retriever is None:
            raise ValueError("graph_retriever must be a valid GraphRetriever instance.")

        if vector_retriever is None:
            raise ValueError("vector_retriever must be a valid VectorRetriever instance.")

        self.graph_retriever = graph_retriever
        self.vector_retriever = vector_retriever

    @staticmethod
    def _deduplicate_vector_chunks(
        vector_chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        seen_content = set()
        deduplicated = []

        for chunk in vector_chunks:
            content = chunk.get("content", "")
            if content not in seen_content:
                seen_content.add(content)
                deduplicated.append(chunk)

        return deduplicated

    @staticmethod
    def _remove_vector_chunks_covered_by_graph(
        vector_chunks: List[Dict[str, Any]], graph_context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        graph_text_values = set()

        for node in graph_context.get("nodes", []):
            for value in node.get("properties", {}).values():
                if isinstance(value, str):
                    graph_text_values.add(value.strip())

        for relationship in graph_context.get("relationships", []):
            for value in relationship.get("properties", {}).values():
                if isinstance(value, str):
                    graph_text_values.add(value.strip())

        filtered_chunks = [
            chunk
            for chunk in vector_chunks
            if chunk.get("content", "").strip() not in graph_text_values
        ]

        return filtered_chunks

    def _merge_context(
        self, graph_context: Dict[str, Any], vector_context: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        deduplicated_chunks = self._deduplicate_vector_chunks(vector_context)
        deduplicated_chunks = self._remove_vector_chunks_covered_by_graph(
            deduplicated_chunks, graph_context
        )

        return {
            "nodes": graph_context.get("nodes", []),
            "relationships": graph_context.get("relationships", []),
            "chunks": deduplicated_chunks,
        }

    def retrieve(self, question: str) -> Dict[str, Any]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")

        try:
            graph_context = self.graph_retriever.retrieve(question)
        except Exception as error:
            raise RuntimeError(f"Graph retrieval failed: {error}") from error

        try:
            vector_context = self.vector_retriever.retrieve(question)
        except Exception as error:
            raise RuntimeError(f"Vector retrieval failed: {error}") from error

        merged_context = self._merge_context(graph_context, vector_context)

        logger.info(
            "Hybrid retrieval complete: %d node(s), %d relationship(s), %d chunk(s) after merge.",
            len(merged_context["nodes"]),
            len(merged_context["relationships"]),
            len(merged_context["chunks"]),
        )

        return {
            "graph_context": graph_context,
            "vector_context": vector_context,
            "merged_context": merged_context,
        }