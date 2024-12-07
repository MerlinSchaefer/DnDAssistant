import logging

import duckdb
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core import vectorstores

from src.llm.deployments import AvailableEmbeddingModels
from src.llm.deployments._get_embedding_model import get_embedding_model
from src.parsing._duckdb.vectorstore import CustomDuckDB

# Basic Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def get_duckdb_retriever(
    db_path: str,
    k: int = 3,
    filter_expression: str | None = None,
    embedding_model: HuggingFaceEmbeddings = None,
) -> vectorstores.VectorStoreRetriever:
    """
    Returns a retriever object configured for document retrieval,
    leveraging the vector_store instance's
    capability to transform into a retriever.

    Returns:
         A retriever object ready for document retrieval operations.
    """
    if not embedding_model:
        embedding = get_embedding_model(model_name=AvailableEmbeddingModels.BGE_LARGE_EN)
    if not filter_expression:
        filter_expression = "TRUE"
    con = duckdb.connect(
        database=db_path,
        read_only=False,
    ).cursor()

    vector_store = CustomDuckDB(
        connection=con,
        embedding=embedding,
        vector_key="embedding",
        id_key="id",
        text_key="text",
        table_name="embeddings",
    )

    return vector_store.as_retriever(search_kwargs={"k": k, "filter": filter_expression})


def get_duckdb_vectorstore(
    db_path: str,
    embedding_model: HuggingFaceEmbeddings = None,
) -> CustomDuckDB:
    """
    Returns a CustomDuckDB object configured for document addition.

    Returns:
         A CustomDuckDB object ready for document operations.
    """
    if not embedding_model:
        embedding = get_embedding_model(model_name=AvailableEmbeddingModels.BGE_LARGE_EN)
    con = duckdb.connect(
        database=db_path,
        read_only=False,
    ).cursor()

    return CustomDuckDB(
        connection=con,
        embedding=embedding,
        vector_key="embedding",
        id_key="id",
        text_key="text",
        table_name="embeddings",
    )
