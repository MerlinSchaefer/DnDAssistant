from langchain.prompts import PromptTemplate

general_prompt = PromptTemplate(
    template="""You are a helpful assistant.
    Follow the users instructions and provide the best answer you
    Try to answer the question based on the chat history first then use the documents.
    If you don't know the answer, just say that you don't know.
    Think before you answer.
    If necessary use the following documents to answer the question.
    User Input: {question}
    Documents: {documents}

    Answer:
    """,
    input_variables=["question", "documents",],
)
