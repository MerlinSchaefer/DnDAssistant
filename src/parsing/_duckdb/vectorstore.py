# Define a new subclass that inherits from DuckDB for filtering
import json
import logging
import re
import uuid
from typing import Any, Iterable

import pandas as pd
from langchain_community.vectorstores import DuckDB
from langchain_core.documents.base import Document

DEFAULT_VECTOR_KEY = "embedding"
DEFAULT_ID_KEY = "id"
DEFAULT_TEXT_KEY = "text"
DEFAULT_TABLE_NAME = "embeddings"
DEFAULT_DOCUMENT_ID_COLUMN = "reference"
SIMILARITY_ALIAS = "similarity_score"


class InvalidFilterExpressionException(Exception):
    """Exception raised for invalid filter expressions."""

    def __init__(self, expression: str, message: str = "Invalid filter expression"):
        self.expression = expression
        self.message = message
        super().__init__(f"{message}: {expression}")


class CustomDuckDB(DuckDB):
    def __init__(self, *args, document_id_column=DEFAULT_DOCUMENT_ID_COLUMN, **kwargs):
        """Initialize CustomDuckDB with custom document ID column.

        Args:
            document_id_column: The column name for document reference.
            *args: Variable length argument list for parent class.
            **kwargs: Arbitrary keyword arguments for parent class.
        """
        # add custom column for filtering, different from PK id in duckdb
        # this column identifies a complete document not chunks
        self._document_id_column = document_id_column
        # initialize the parent class with other arguments
        super().__init__(*args, **kwargs)

    # added file_name column for retrieval filter
    # rest is langchain code
    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: list[dict] | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Turn texts into embedding and add it to the database
            using a pandas DataFrame.

        Args:
            texts: Iterable of strings to add to the vectorstore.
            metadatas: Optional list of metadatas associated with the texts.
            kwargs: Additional parameters including optional 'ids' to associate
              with the texts.

        Returns:
            List of ids of the added texts.
        """

        # Extract ids from kwargs or generate new ones if not provided
        ids = kwargs.pop("ids", [str(uuid.uuid4()) for _ in texts])

        # Embed texts and create documents
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        # embeddings = self._embedding.embed_documents(list(texts))
        data = []
        for idx, text in enumerate(texts):
            embedding = self._embedding.embed_query(text)
            # embedding = embeddings[idx]
            # Serialize metadata if present, else default to None
            metadata = json.dumps(metadatas[idx]) if metadatas and idx < len(metadatas) else None
            # Extract file_name from metadata if present
            reference = (
                metadatas[idx].get(self._document_id_column)
                if metadatas and idx < len(metadatas) and self._document_id_column in metadatas[idx]
                else None
            )
            data.append(
                {
                    self._id_key: ids[idx],
                    self._text_key: text,
                    self._vector_key: embedding,
                    "metadata": metadata,
                    self._document_id_column: reference,
                }
            )

        # noinspection PyUnusedLocal
        df = pd.DataFrame.from_dict(data)  # noqa: F841
        self._connection.execute(
            f"INSERT INTO {self._table_name} SELECT * FROM df",
        )
        return ids

    # simple validation check with custom col file_name
    def _validate_filter_expression(self, filter_expression: str) -> bool:
        """
        Validates if the filter expression is in the format of
        'reference IN (<some list entries>)'.

        Args:
            filter_expression: The filter expression string to validate.

        Returns:
            bool: True if the filter expression matches the required format, otherwise False.
        """
        # Define the regular expression pattern
        pattern = r"^reference\s+IN\s+\(\s*(?:\'[^\']*\'\s*,\s*)*\'[^\']*\'\s*\)$"

        # Check if the filter expression matches the pattern
        return bool(re.match(pattern, filter_expression.strip()))

    def similarity_search(self, query: str, k: int = 4, **kwargs: Any) -> list[Document]:
        """Performs a similarity search for a given query string.

        Args:
            query: The query string to search for.
            k: The number of similar texts to return.

        Returns:
            A list of Documents most similar to the query.
        """
        # Default filter expression that always returns true
        filter_expression = kwargs.get("filter", "TRUE")
        # Validate filter expression
        if filter_expression != "TRUE":
            valid_filter = self._validate_filter_expression(filter_expression)
            if not valid_filter:
                raise InvalidFilterExpressionException(filter_expression)

        logging.info(f"Agent filter expression: {filter_expression}")

        embedding = self._embedding.embed_query(query)  # type: ignore
        list_cosine_similarity = self.duckdb.FunctionExpression(
            "list_cosine_similarity",
            self.duckdb.ColumnExpression(self._vector_key),
            self.duckdb.ConstantExpression(embedding),
        )

        docs = (
            self._table.filter(filter_expression)
            .select(
                *[
                    self.duckdb.StarExpression(exclude=[]),
                    list_cosine_similarity.alias(SIMILARITY_ALIAS),
                ]
            )
            .order(f"{SIMILARITY_ALIAS} desc")
            .limit(k)
            .fetchdf()
        )
        return [
            Document(
                page_content=docs[self._text_key][idx],
                metadata={
                    **json.loads(docs["metadata"][idx]),
                    # using underscore prefix to avoid conflicts with user metadata keys
                    f"_{SIMILARITY_ALIAS}": docs[SIMILARITY_ALIAS][idx],
                }
                if docs["metadata"][idx]
                else {},
            )
            for idx in range(len(docs))
        ]

    def delete_document(self, doc_reference: str) -> None:
        """Deletes a document from the database by its ID.

        Args:
            doc_reference: The document reference (not ID) of the document to delete.
        """
        self._connection.execute(f"DELETE FROM {self._table_name} WHERE {self._document_id_column} = '{doc_reference}'")

    # Overwrites default langchain implementation which requires write permissions
    def _ensure_table(self) -> None:
        """Ensures the table for storing embeddings exists."""
        if not self._table_exists():  # type: ignore # mypy thinks method is truthy
            try:
                self._create_table_if_not_exists()
            except Exception as e:
                logging.error(e)
                logging.error("Connection possibly read-only when write required.")

    def _table_exists(self) -> bool:
        """Checks if table exists, compliant with read_only mode of retriever."""
        query = f"SELECT COUNT(*) FROM information_schema.tables WHERE table_name = '{self._table_name}'"
        result = self._connection.execute(query).fetchall()
        return len(result) > 0 and result[0][0] > 0

    def _create_table_if_not_exists(self) -> None:
        """Creates table for storing embeddings if non-existent. Not possible with retriever."""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self._table_name} (
            {self._id_key} VARCHAR PRIMARY KEY,
            {self._text_key} VARCHAR,
            {self._vector_key} FLOAT[],
            metadata VARCHAR,
            {self._document_id_column} VARCHAR
        )
        """
        self._connection.execute(create_table_sql)
