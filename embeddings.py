from typing import List

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingGenerator:

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        self.model_name = model_name

        try:
            self.embedding_model = HuggingFaceEmbeddings(model_name=self.model_name)
        except Exception as error:
            raise RuntimeError(
                f"Failed to load embedding model '{self.model_name}': {error}"
            ) from error

    def embed_text(self, text: str) -> List[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("embed_text requires a non-empty string.")

        try:
            return self.embedding_model.embed_query(text)
        except Exception as error:
            raise RuntimeError(f"Failed to generate embedding for text: {error}") from error

    def embed_documents(self, chunks: List[str]) -> List[List[float]]:
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("embed_documents requires a non-empty list of strings.")

        for chunk in chunks:
            if not isinstance(chunk, str) or not chunk.strip():
                raise ValueError("All items in chunks must be non-empty strings.")

        try:
            return self.embedding_model.embed_documents(chunks)
        except Exception as error:
            raise RuntimeError(f"Failed to generate embeddings for chunks: {error}") from error