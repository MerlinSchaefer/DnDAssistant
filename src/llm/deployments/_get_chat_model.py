from enum import Enum

from langchain_community.chat_models import ChatOllama


class AvailableChatModels(Enum):
    LLAMA_3_2 = "llama3.2"
    LLAMA_3_1 = "llama3.1"
    GEMMA_2 = "gemma2"


def get_chat_model(
    model_name: AvailableChatModels = AvailableChatModels.LLAMA_3_2,
    temperature: float = 0.5,
    context_length: int = 16000,
) -> ChatOllama:
    return ChatOllama(model=model_name.value, temperature=temperature, num_ctx=context_length)
