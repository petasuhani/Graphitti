import os
from typing import Any, Dict, List, Optional, TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

from graph_retriever import GraphRetriever
from embeddings import EmbeddingGenerator
from chroma import ChromaManager
from vector_retriever import VectorRetriever
from hybrid_retriever import HybridRetriever
from answer_generator import AnswerGenerator

load_dotenv()


class GraphState(TypedDict):
    question: str
    graph_context: Optional[Dict[str, Any]]
    vector_context: Optional[List[Dict[str, Any]]]
    merged_context: Optional[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]


_embedding_generator = EmbeddingGenerator()
_chroma_manager = ChromaManager(_embedding_generator)
_vector_retriever = VectorRetriever(_chroma_manager)


_graph_retriever = GraphRetriever(
    uri=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    database=os.getenv("NEO4J_DATABASE"),
)

_hybrid_retriever = HybridRetriever(_graph_retriever, _vector_retriever)
_answer_generator = AnswerGenerator()


def retrieve_node(state: GraphState) -> GraphState:
    result = _hybrid_retriever.retrieve(state["question"])
    return {
        "graph_context": result["graph_context"],
        "vector_context": result["vector_context"],
        "merged_context": result["merged_context"],
    }


def generate_answer_node(state: GraphState) -> GraphState:
    result = _answer_generator.generate(
        state["question"],
        state.get("graph_context"),
        state.get("vector_context"),
        state.get("merged_context"),
    )
    return {
        "answer": result["answer"],
        "sources": result.get("sources", []),
    }


builder = StateGraph(GraphState)

builder.add_node("retrieve_node", retrieve_node)
builder.add_node("generate_answer_node", generate_answer_node)

builder.add_edge(START, "retrieve_node")
builder.add_edge("retrieve_node", "generate_answer_node")
builder.add_edge("generate_answer_node", END)

graph = builder.compile()