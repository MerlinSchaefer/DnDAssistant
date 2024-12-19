from datetime import datetime

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.vectorstores import VectorStoreRetriever

from src.llm.memory import get_or_create_memory
from src.llm.prompts.assistant import general_prompt


def get_rag_chain(_llm: ChatOllama, prompt: PromptTemplate = general_prompt, with_history: bool = False):
    rag_chain = prompt | _llm | StrOutputParser()
    if with_history:
        rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_or_create_memory,
            input_messages_key="question",
        )
    return rag_chain


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
