from langchain import tools as lc_tools
from langchain_core import retrievers as lcc_retrievers


def get_document_retrieval_tool(
    retriever: lcc_retrievers.BaseRetriever,
) -> lc_tools.Tool:
    """Creates a tool for retrieving chunks from multiple documents.

    Args:
        retriever: The retriever to use for retrieving documents.
    """
    tool = lc_tools.retriever.create_retriever_tool(
        retriever=retriever,  # type: ignore
        name="vector_store_document_retriever",
        description="""Use this to retrieve documents from the vectorstore.
        These will be documents the users have uploaded, such as texts about DnD Settings, Characters, Notes.""",
    )

    return tool
