import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory

# from src.db import get_vector_store, get_storage_context
from src.llm.deployments import (
    AvailableChatModels,
    AvailableEmbeddingModels,
    get_chat_model,
    get_embedding_model,
)
from src.llm.memory import create_memory
from src.llm.prompts.assistant import general_prompt
from src.parsing import DocumentProcessor, get_duckdb_retriever, get_duckdb_vectorstore

# Configure basic debug logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# Funcs to create page resources
@st.cache_resource
def load_llm(model_name=AvailableChatModels.LLAMA_3_2) -> ChatOllama:
    return get_chat_model(model_name=model_name)


@st.cache_resource
def load_emb() -> HuggingFaceEmbeddings:
    return get_embedding_model(model_name=AvailableEmbeddingModels.BGE_LARGE_EN)  # BGE_LARGE_EN


@st.cache_resource
def load_vectorstore(db_path: str = "./data/duckdb_embeddings"):
    # return SKLearnVectorStore(embedding=load_emb(), n_neighbors=1)
    return get_duckdb_vectorstore(db_path=db_path)


@st.cache_resource
def load_retriever(db_path: str = "./data/duckdb_embeddings"):
    return get_duckdb_retriever(
        db_path=db_path,
        k=3,
        filter_expression=None,
    )


# ONLY FOR TESTING
class RAGApplication:
    def __init__(self, retriever, rag_chain, session_id=None):
        self.retriever = retriever
        self.rag_chain = rag_chain
        if not session_id:
            self.session_id = datetime.now().strftime("%Y%m%d%H%M%S")

    def invoke(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke(
            {"question": question, "documents": doc_texts}, config={"configurable": {"session_id": self.session_id}}
        )
        return answer

    def _test(self):
        return self.invoke("Hello, how are you?")


@st.cache_resource
def get_rag_app(_llm: ChatOllama, prompt: PromptTemplate = general_prompt):
    retriever = load_retriever()
    rag_chain = prompt | _llm | StrOutputParser()
    logging.debug(f"RAG chain: {rag_chain}")
    rag_chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        create_memory,  # get_or_create_memory, NEEDS FIX FOR RETRIEVAL OF .messages
        input_messages_key="question",
    )
    print(f"RAG chain with history: {rag_chain_with_history}")
    # Define the RAG application class

    rag_app = RAGApplication(retriever, rag_chain_with_history)
    print(f"RAG app: {rag_app}")

    return rag_app


# Set page configuration
st.set_page_config(page_title="Dungeon Master Assistant", layout="centered")
# Title at the top of the page
st.title("Dungeon Master Assistant Chat")
# Sidebar for additional options
st.sidebar.title("Options")
st.sidebar.write("Manage models, documents, and queries.")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "retriever" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()
if "rag_pp" not in st.session_state:
    st.session_state.query_engine = None


# Sidebar: Model selection
# model_name_str = st.sidebar.selectbox("Select a chat model", options=[model.name for model in AvailableChatModels])

# Sidebar: Button to load the model and query engine
load_model_button = st.sidebar.button("Load Model and Engine")

# Sidebar: File Upload for Document Parsing
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "txt", "md"])
process_file_button = st.sidebar.button("Process File")


# Process the uploaded document
if process_file_button and uploaded_file:
    # Save the uploaded file to a temporary path
    temp_file_path = Path(f"temp_{uploaded_file.name}")
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Parse and embed the document
    st.write(f"Processing document: {uploaded_file.name}...")
    document_parser = DocumentProcessor()
    try:
        docs = document_parser.load_and_process_documents(str(temp_file_path))
        st.success("Document processed into doc chunks successfully!")

        # Add docs to the vector store

        st.session_state.vectorstore.add_documents(docs)
        st.success("Docs added to the vector store!")

        # Optionally, display the docs
        st.write(docs)
        # st.write("Retrieved dpcs:")
        # doc_ids = [doc.id for doc in docs]
        # docs_retrieved = vectorstore.get_by_ids(doc_ids)
        # for node in nodes_retrieved:
        #     st.write(node)

    except Exception as e:
        st.error(f"Error processing document: {e}")
    finally:
        # Clean up the temporary file
        temp_file_path.unlink()

if load_model_button:
    # model_name = AvailableChatModels[model_name_str]  # Convert string to Enum
    llm = load_llm()
    st.session_state.llm = llm
    st.session_state.rag_app = get_rag_app(st.session_state.llm)
    # st.session_state.query_engine = get_query_engine(st.session_state.llm)
    st.success("Model and query engine loaded successfully!")

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input field for asking questions
if prompt := st.chat_input("Ask me anything"):
    # Display user's message in the chat and add to session state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get the response from the query engine if it's loaded
    # if st.session_state.query_engine:

    # response = st.session_state.query_engine.query(prompt)
    # print(response.source_nodes)
    if st.session_state.rag_app:
        try:
            response = st.session_state.rag_app.invoke(prompt)
        except Exception as e:
            response = f"Error: {e}"
    else:
        response = "Model and query engine are not loaded. Please load them from the sidebar."

    # Add assistant's response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
