from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


class QueryClassification(BaseModel):
    label: Literal["GRAPH", "VECTOR", "HYBRID"] = Field(
        description=(
            "The retrieval strategy for the question. "
            "GRAPH for factual/entity/relationship lookups over structured "
            "knowledge. VECTOR for explanations, definitions, descriptions, "
            "or summarization. HYBRID for comparisons, reasoning, timelines, "
            "or multi-hop questions needing both graph facts and text context."
        )
    )


class QueryClassifier:

    _VALID_LABELS = ("GRAPH", "VECTOR", "HYBRID")

    def __init__(self, llm: BaseChatModel | None = None) -> None:
        if llm is not None:
            self.llm = llm
        else:
            try:
                self.llm = ChatOllama(model="llama3", temperature=0)
            except Exception as error:
                raise RuntimeError(f"Failed to initialize default LLM: {error}") from error

        self.parser = PydanticOutputParser(pydantic_object=QueryClassification)

        try:
            self.structured_llm = self.llm.with_structured_output(QueryClassification)
        except Exception:
            self.structured_llm = None

        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a query classifier for a knowledge-graph question "
                    "answering system. Classify the user's question into exactly "
                    "one category:\n\n"
                    "GRAPH - factual questions, entity lookup, relationship "
                    "lookup, structured knowledge.\n"
                    "VECTOR - explanations, definitions, descriptive questions, "
                    "summarization.\n"
                    "HYBRID - comparisons, reasoning, timelines, multi-hop "
                    "questions, or questions needing both graph facts and "
                    "textual context.\n\n"
                    "Respond with only the category label.",
                ),
                ("human", "{question}"),
            ]
        )

    def classify(self, question: str) -> str:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string.")

        try:
            if self.structured_llm is not None:
                chain = self.prompt | self.structured_llm
                result: QueryClassification = chain.invoke({"question": question})
                label = result.label
            else:
                fallback_prompt = self.prompt.partial(
                    format_instructions=self.parser.get_format_instructions()
                )
                chain = fallback_prompt | self.llm | self.parser
                result = chain.invoke({"question": question})
                label = result.label
        except Exception as error:
            raise RuntimeError(f"Query classification failed: {error}") from error

        if label not in self._VALID_LABELS:
            raise RuntimeError(f"Classifier returned an invalid label: {label}")

        return label