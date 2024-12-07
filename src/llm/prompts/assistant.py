from langchain.prompts import PromptTemplate

general_prompt = PromptTemplate(
    template="""You are a helpful assistant.
    Follow the users instructions and provide the best answer you can.
    If necessary answer the question with the help of the documents or the chat history.
    If you don't know the answer, just say that you don't know.
    Think before you answer.

    User Input: {question}
    Documents: {documents}

    Answer:
    """,
    input_variables=[
        "question",
        "documents",
    ],
)
