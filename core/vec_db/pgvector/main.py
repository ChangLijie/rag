import os
from typing import List

from haystack.components.writers import DocumentWriter
from haystack_integrations.components.retrievers.pgvector import (
    PgvectorEmbeddingRetriever,
)
from haystack_integrations.document_stores.pgvector import PgvectorDocumentStore

os.environ["PG_CONN_STR"] = (
    f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}@host.docker.internal:{os.environ['POSTGRES_PORT']}/{os.environ['POSTGRES_DB']}"
)


class Operator:
    """Pgvector operator"""

    def __init__(
        self,
        recreate_table: bool = False,
        embedding_dimension: int = 384,
        vector_function: str = "cosine_similarity",
        search_strategy: str = "hnsw",
    ) -> None:
        # Initializing the DocumentStore
        self.vector_function = vector_function
        self.document_store = PgvectorDocumentStore(
            embedding_dimension=embedding_dimension,
            vector_function=self.vector_function,
            recreate_table=recreate_table,
            search_strategy=search_strategy,
        )

        self.document_writer = DocumentWriter(self.document_store)
        self.set_retriever()

    def save(self, documents: list) -> None:
        """
        Save document to db.

        Args:
            documents (list): data after document_embedding.
                More information :https://docs.haystack.deepset.ai/reference/document-writers-api#documentwriter
        """
        self.document_writer.run(documents=documents)

    def set_retriever(self, top_k: int = 2) -> None:
        """
        Set retriever.

        Args:
            top_k (int, optional): Maximum number of result. Defaults to 2.
        """
        self.retriever = PgvectorEmbeddingRetriever(
            document_store=self.document_store,
            top_k=top_k,
            vector_function=self.vector_function,
        )

    def search(
        self,
        query_embedding: List[float],
        filters: dict = {
            "operator": "AND",
            "conditions": [
                {"field": "meta.privacy", "operator": "!=", "value": "1"},
            ],
        },
    ) -> List[float]:
        """
        Retriever from vector database.
            More information :https://docs.haystack.deepset.ai/reference/integrations-pgvector#pgvectorembeddingretriever


        Args:
            query_embedding (List[float]): Prompt after embedding .
            filters (dict, optional): filter . Defaults to { "operator": "AND", "conditions": [ {"field": "meta.privacy", "operator": "!=", "value": "1"}, ], }.

        Returns:
            List[float]: retriever result.
        """
        retriever_result = self.retriever.run(
            query_embedding=query_embedding, filters=filters
        )
        return retriever_result
