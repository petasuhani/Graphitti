import hashlib
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

from embeddings import EmbeddingGenerator


class _EmbeddingGeneratorAdapter(Embeddings):

    def __init__(self, embedding_generator: EmbeddingGenerator) -> None:
        self._embedding_generator = embedding_generator

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embedding_generator.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embedding_generator.embed_text(text)


class ChromaManager:

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        persist_directory: str = "./chroma_db",
        collection_name: str = "graphitti_documents",
    ) -> None:
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.embedding_generator = embedding_generator

        embedding_function = _EmbeddingGeneratorAdapter(self.embedding_generator)

        try:
            self.vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=embedding_function,
                persist_directory=self.persist_directory,
            )
        except Exception as error:
            raise RuntimeError(f"Failed to initialize ChromaDB vector store: {error}") from error

    @staticmethod
    def _make_chunk_id(chunk: str, metadata: Dict[str, Any]) -> str:
        source = str(metadata.get("url", ""))
        hash_input = f"{source}::{chunk}".encode("utf-8")
        return hashlib.sha256(hash_input).hexdigest()

    def add_documents(self, chunks: List[str], metadata: List[Dict[str, Any]]) -> None:
        if not isinstance(chunks, list) or not chunks:
            raise ValueError("chunks must be a non-empty list of strings.")

        if not isinstance(metadata, list) or not metadata:
            raise ValueError("metadata must be a non-empty list of dictionaries.")

        if len(chunks) != len(metadata):
            raise ValueError("chunks and metadata must have the same length.")

        try:
            documents = [
                Document(page_content=chunk, metadata=meta)
                for chunk, meta in zip(chunks, metadata)
            ]
            ids = [
                self._make_chunk_id(chunk, meta)
                for chunk, meta in zip(chunks, metadata)
            ]

            self.vectorstore.add_documents(documents, ids=ids)
        except Exception as error:
            raise RuntimeError(f"Failed to add documents to ChromaDB: {error}") from error

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string.")

        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer.")

        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as error:
            raise RuntimeError(f"Similarity search failed: {error}") from error

    def delete_collection(self) -> None:
        try:
            self.vectorstore.delete_collection()
        except Exception as error:
            raise RuntimeError(f"Failed to delete ChromaDB collection: {error}") from error

    def get_collection_count(self) -> int:
        try:
            return self.vectorstore._collection.count()
        except Exception as error:
            raise RuntimeError(f"Failed to get collection count: {error}") from error