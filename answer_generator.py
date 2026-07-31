import logging
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AnswerGenerator:
    def __init__(self, llm: Optional[BaseChatModel] = None) -> None:
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = ChatOllama(model="llama3", temperature=0)
            except Exception as error:
                raise RuntimeError(f"Failed to initialize default LLM: {error}") from error

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the answer-generation component of a knowledge "
                    "graph question-answering system called Graphitti. "
                    "You must answer the user's question using ONLY the "
                    "context provided below.\n\n"
                    "STRICT RULES — follow these exactly:\n"
                    "1. Do NOT use any fact, name, number, date, or detail "
                    "that is not explicitly present in the context below.\n"
                    "2. If the context only partially answers the question, "
                    "answer only with the available information.\n"
                    "3. If the context does not contain enough information, "
                    "respond exactly with: 'I don't have enough information in the knowledge graph to answer that.'\n"
                    "4. Every fact must be supported by the provided context.\n\n"
                    "GRAPH CONTEXT:\n{graph_context_text}\n\n"
                    "VECTOR CONTEXT:\n{vector_context_text}",
                ),
                ("human", "{question}"),
            ]
        )

    @staticmethod
    def _format_graph_context(graph_context: Optional[Dict[str, Any]]) -> str:
        if not graph_context:
            return "No graph facts were retrieved."

        nodes = graph_context.get("nodes", [])
        relationships = graph_context.get("relationships", [])

        if not nodes and not relationships:
            return "No graph facts were retrieved."

        lines: List[str] = []

        if nodes:
            lines.append("Entities:")
            for node in nodes:
                properties = node.get("properties", {})
                name = properties.get("name", node.get("id", "unknown"))
                labels = ", ".join(node.get("labels", []))
                lines.append(
                    f"- {name} (labels: {labels}), properties: {properties}"
                )

        if relationships:
            lines.append("Relationships:")
            for relationship in relationships:
                lines.append(
                    f"- {relationship.get('start_node_id')} "
                    f"-[{relationship.get('type')}]-> "
                    f"{relationship.get('end_node_id')}, "
                    f"properties: {relationship.get('properties', {})}"
                )

        return "\n".join(lines)

    @staticmethod
    def _format_vector_context(
        vector_context: Optional[List[Dict[str, Any]]]
    ) -> str:
        if not vector_context:
            return "No vector context was retrieved."

        lines: List[str] = []

        for index, chunk in enumerate(vector_context, start=1):
            content = chunk.get("content", "")
            lines.append(f"[{index}] {content}")

        return "\n\n".join(lines)

    @staticmethod
    def _build_sources(
        graph_context: Optional[Dict[str, Any]],
        vector_context: Optional[List[Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        sources: List[Dict[str, Any]] = []

        if graph_context:
            for node in graph_context.get("nodes", []):
                properties = node.get("properties", {})
                sources.append(
                    {
                        "type": "graph",
                        "id": node.get("id"),
                        "name": properties.get("name"),
                        "labels": node.get("labels", []),
                    }
                )

        if vector_context:
            for chunk in vector_context:
                sources.append(
                    {
                        "type": "vector",
                        "metadata": chunk.get("metadata", {}),
                    }
                )

        return sources

    def generate(
        self,
        question: str,
        graph_context: Optional[Dict[str, Any]],
        vector_context: Optional[List[Dict[str, Any]]],
        merged_context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:

        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")

        graph_context_text = self._format_graph_context(graph_context)
        vector_context_text = self._format_vector_context(vector_context)
        sources = self._build_sources(graph_context, vector_context)

        try:
            chain = self.prompt | self.llm
            response = chain.invoke(
                {
                    "question": question,
                    "graph_context_text": graph_context_text,
                    "vector_context_text": vector_context_text,
                }
            )
            answer_text = response.content.strip()

        except Exception as error:
            logger.error("Answer generation failed: %s", error)
            raise RuntimeError(
                f"Failed to generate answer: {error}"
            ) from error

        return {
            "answer": answer_text,
            "sources": sources,
        }