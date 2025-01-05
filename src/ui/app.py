import logging
from datetime import datetime
from pathlib import Path

import mlflow
import streamlit as st
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings

from src.config import config
from src.evaluation.logging_utils import log_chat
from src.llm.agents import ConversationalRetrievalAgentWithMemory
from src.llm.deployments import (
    AvailableChatModels,
    AvailableEmbeddingModels,
    get_chat_model,
    get_embedding_model,
)
from src.llm.memory import PickleMemory, list_memories
from src.parsing import DocumentProcessor, get_duckdb_retriever, get_duckdb_vectorstore

# Configure basic debug logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

mlflow.set_experiment("App Chat")
# Set the chat log path for MLFlow
if "mlflow_chatlog_path" not in st.session_state:
    st.session_state.mlflow_chatlog_path = (
        f"{config.default_chatlog_path}/mlflow_logging/chatlog_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"
    )


@st.cache_resource
def load_llm(
    model_name: AvailableChatModels = config.app_chat_model,
    temperature: float = config.app_chat_temperature,
    context_length: int = config.app_chat_context_length,
) -> ChatOllama:
    return get_chat_model(model_name=model_name, temperature=temperature, context_length=context_length)


@st.cache_resource
def load_emb(
    model_name: AvailableEmbeddingModels = config.app_embedding_model,  # BGE_LARGE_EN
) -> HuggingFaceEmbeddings:
    return get_embedding_model(model_name=model_name)  # BGE_LARGE_EN


@st.cache_resource
def load_vectorstore(db_path: str = "./data/duckdb_embeddings"):
    return get_duckdb_vectorstore(db_path=db_path)


@st.cache_resource
def load_retriever(db_path: str = "./data/duckdb_embeddings"):
    return get_duckdb_retriever(
        db_path=db_path,
        k=5,
        filter_expression=None,
    )


@st.cache_resource
def load_memory(session_id: str = datetime.now().strftime("%Y%m%d%H%M%S"), storage_path: str | None = None):
    memory = PickleMemory(session_id=session_id, storage_path=storage_path)
    memory.load()
    return memory


@st.cache_resource
def get_rag_app(_llm: ChatOllama, _memory: PickleMemory) -> ConversationalRetrievalAgentWithMemory:
    retriever = load_retriever()
    rag_app = ConversationalRetrievalAgentWithMemory(
        retriever=retriever,
        chat_model=_llm,
        memory=_memory.get_memory(),
    )
    return rag_app


# Set page configuration
st.set_page_config(page_title="Dungeon Master Assistant", layout="centered")
# Title at the top of the page
st.title("Dungeon Master Assistant Chat")
# Sidebar for additional options
st.sidebar.title("Options")
st.sidebar.write("Manage models, documents, and memories.")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
if "llm" not in st.session_state:
    st.session_state.llm = None
if "retriever" not in st.session_state:
    st.session_state.vectorstore = load_vectorstore()
if "memory" not in st.session_state:
    st.session_state.memory = load_memory()
if "rag_app" not in st.session_state:
    st.session_state.rag_app = None


# Sidebar: Model selection
model_name_str = st.sidebar.selectbox("Select a chat model", options=[model.name for model in AvailableChatModels])

# Sidebar: Button to load the model and query engine
load_model_button = st.sidebar.button("Load Model and Engine")

# Sidebar: File Upload for Document Parsing
uploaded_files = st.sidebar.file_uploader("Upload a document", type=["pdf", "txt", "md"], accept_multiple_files=True)
process_file_button = st.sidebar.button("Process File")

available_memories = st.sidebar.selectbox(label="Chats", options=list_memories())

# Sidebar: Button to list available documents
list_docs_button = st.sidebar.button("List Documents")

if list_docs_button:
    st.sidebar.write(doc.metadata["source"] for doc in st.session_state.vectorstore.list_all_documents())

# Process the uploaded document
if process_file_button and uploaded_files:
    for uploaded_file in uploaded_files:
        # Save the uploaded file to a temporary path
        temp_file_path = Path(f"temp_{uploaded_file.name}")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Parse and embed the document
        st.sidebar.write(f"Processing document: {uploaded_file.name}...")
        document_parser = DocumentProcessor()
        try:
            docs = document_parser.load_and_process_documents(str(temp_file_path))
            st.sidebar.success("Document processed into doc chunks successfully!")

            # Add docs to the vector store

            st.session_state.vectorstore.add_documents(docs)
            st.sidebar.success("Docs added to the vector store!")

            # Optionally, display the docs
            st.sidebar.write(docs)

        except Exception as e:
            st.error(f"Error processing document: {e}")
        finally:
            # Clean up the temporary file
            temp_file_path.unlink()

if load_model_button and model_name_str:
    with mlflow.start_run() as run:
        st.session_state.mlflow_run_id = run.info.run_id
        mlflow.log_param("selected_model", model_name_str)

        model_name = AvailableChatModels[model_name_str]  # Convert string to Enum
        llm = load_llm(model_name=model_name)
        st.session_state.llm = llm
        st.session_state.rag_app = get_rag_app(st.session_state.llm, st.session_state.memory)
        # log the langchain model to mlflow
        mlflow.langchain.log_model(st.session_state.rag_app._agent_executor, "react agent")

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

    # Save the chat messages to the memory
    st.session_state.memory.save()

    # Log the chat to MLFlow
    mlflow.start_run(run_id=st.session_state.mlflow_run_id)

    log_chat(prompt, response, log_file=st.session_state.mlflow_chatlog_path)
    mlflow.end_run()
