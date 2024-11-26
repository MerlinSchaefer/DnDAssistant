import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_community.vectorstores import SKLearnVectorStore
from langchain_core.output_parsers import StrOutputParser

# from src.db import get_vector_store, get_storage_context
# from src.llm.deployments import AvailableChatModels, get_chat_model
# from src.llm.embeddings import AvailableEmbeddingModels, get_embedding_model
# from src.llm.retriever import VectorDBRetriever
# from src.parsers import DocumentParser
from src.llm.prompts.assistant import general_prompt


# Funcs to create page resources
@st.cache_resource
def load_llm(model_name=None) -> ChatOllama:
    # return get_chat_model(
    #     model_name=model_name,
    # )
    local_llm = "llama3.2"
    return ChatOllama(model=local_llm, temperature=0.5, num_ctx=16000)


# @st.cache_resource
# def load_emb() -> HuggingFaceEmbeddings:
#     return get_embedding_model(model_name=AvailableEmbeddingModels.BGE_LARGE_EN)  # BGE_LARGE_EN


@st.cache_resource
def load_vector_store():
    return SKLearnVectorStore(
        embedding=HuggingFaceEmbeddings(model_kwargs={"device": "cuda"})  # model_name ="BAAI/bge-large-en-v1.5",
    )
    # return get_vector_store(
    #     table_name="dev_vectors_2",  # TODO: adjust
    #     embed_dim=1024,  # TODO: adjust
    # )


@st.cache_resource
def load_retriever():
    vector_store = load_vector_store()
    return vector_store.as_retriever(k=3)


# ONLY FOR TESTING
class RAGApplication:
    def __init__(self, retriever, rag_chain):
        self.retriever = retriever
        self.rag_chain = rag_chain

    def run(self, question):
        # Retrieve relevant documents
        documents = self.retriever.invoke(question)
        # Extract content from retrieved documents
        doc_texts = "\\n".join([doc.page_content for doc in documents])
        # Get the answer from the language model
        answer = self.rag_chain.invoke({"question": question, "documents": doc_texts})
        return answer


@st.cache_resource
def get_rag_app(_llm: ChatOllama, prompt: PromptTemplate = general_prompt):
    retriever = load_retriever()
    rag_chain = prompt | _llm | StrOutputParser()
    # Define the RAG application class
    rag_app = RAGApplication(retriever, rag_chain)
    return rag_app


# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []
llm = None
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None


# Set page configuration
st.set_page_config(page_title="Dungeon Master Assistant", layout="centered")
# Title at the top of the page
st.title("Dungeon Master Assistant Chat")
# Sidebar for additional options
st.sidebar.title("Options")
st.sidebar.write("Manage models, documents, and queries.")


# Sidebar: Model selection
# model_name_str = st.sidebar.selectbox("Select a chat model", options=[model.name for model in AvailableChatModels])

# Sidebar: Button to load the model and query engine
load_model_button = st.sidebar.button("Load Model and Engine")

# Sidebar: File Upload for Document Parsing
uploaded_file = st.sidebar.file_uploader("Upload a document", type=["pdf", "txt", "md"])
doc_type = st.sidebar.selectbox("Document type", options=["pdf", "txt", "md"])
process_file_button = st.sidebar.button("Process File")


# Load the model and query engine if the button is clicked
if load_model_button:
    # model_name = AvailableChatModels[model_name_str]  # Convert string to Enum
    llm = load_llm()
    st.session_state.llm = llm
    print(llm.invoke("Hello"))
    # Settings.llm = llm

    # st.session_state.query_engine = get_query_engine(st.session_state.llm)
    st.success("Model and query engine loaded successfully!")

# # Process the uploaded document
# if process_file_button and uploaded_file:
#     # Save the uploaded file to a temporary path
#     temp_file_path = Path(f"temp_{uploaded_file.name}")
#     with open(temp_file_path, "wb") as f:
#         f.write(uploaded_file.getbuffer())

#     # Parse and embed the document
#     st.write(f"Processing document: {uploaded_file.name} as {doc_type}...")
#     document_parser = DocumentParser(temp_file_path, doc_type)
#     try:
#         nodes = document_parser.load_chunk_and_embed()
#         st.success("Document processed into nodes successfully!")

#         # Add nodes to the vector store
#         vectorstore = load_vector_store()
#         vectorstore.add(nodes)
#         st.success("Nodes added to the vector store!")

#         # Optionally, display the nodes
#         st.write("Retrieved nodes:")
#         node_ids = [node.id_ for node in nodes]
#         nodes_retrieved = vectorstore.get_nodes(node_ids)
#         for node in nodes_retrieved:
#             st.write(node)

#     except Exception as e:
#         st.error(f"Error processing document: {e}")
#     finally:
#         # Clean up the temporary file
#         temp_file_path.unlink()


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
    if st.session_state.llm:
        try:
            response = st.session_state.llm.invoke(prompt)
        except Exception as e:
            response = f"Error: {e}"
    else:
        response = "Model and query engine are not loaded. Please load them from the sidebar."

    # Add assistant's response to chat history and display it
    st.session_state.messages.append({"role": "assistant", "content": str(response.content)})
    with st.chat_message("assistant"):
        st.markdown(str(response.content))
