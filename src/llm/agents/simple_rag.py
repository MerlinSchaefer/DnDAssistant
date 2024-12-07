from datetime import datetime

from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever


class RAGApplication:
    def __init__(
        self, retriever: VectorStoreRetriever, rag_chain: RunnableWithMessageHistory, session_id: str | None = None
    ):
        self.retriever = retriever
        self.rag_chain = rag_chain
        if not session_id:
            self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")

    def invoke(self, question: str) -> str:
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke(
            {"question": question, "documents": doc_texts}, config={"configurable": {"session_id": self.session_id}}
        )
        return answer

    def _test(self) -> str:
        return self.invoke("Hello, how are you?")
