from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents.base import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters.base import Language

from src.config import config


class DocumentProcessor:
    def __init__(
        self,
        language_for_splitter: Language = Language.MARKDOWN,
        max_chunk_size: int = config.default_chunk_size,
        max_chunk_overlap: int = config.default_chunk_overlap,
    ):
        """
        Initializes the DocumentProcessor with metadata keys, content key,
        splitter chunk size, and splitter overlap size.

        Args:
            language_for_splitter: The language to use for the text splitter.
            max_chunk_size: The maximum chunk size for the text splitter.
            max_chunk_overlap: The maximum chunk overlap for the text splitter.
        """
        self.language = language_for_splitter
        self.max_chunk_size = max_chunk_size
        self.max_chunk_overlap = max_chunk_overlap

    def load_and_process_documents(self, file_path: str) -> list[Document]:
        """
        Load documents from a file and process them.

        Args:
            file_path: The path to the file containing the documents.

        Returns:
            The list of processed documents.
        """
        # possibly use different loader depending on what we use for input
        if file_path.endswith(".md"):
            loader = UnstructuredMarkdownLoader(file_path)
        elif file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        else:
            loader = TextLoader(file_path)  # type: ignore
        documents = loader.load()

        documents = RecursiveCharacterTextSplitter.from_language(
            language=self.language,
            chunk_size=self.max_chunk_size,
            chunk_overlap=self.max_chunk_overlap,
        ).split_documents(documents)

        return documents
