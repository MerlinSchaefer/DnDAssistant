from enum import Enum

from langchain_community.embeddings import HuggingFaceEmbeddings


class AvailableEmbeddingModels(Enum):
    BGE_LARGE_EN = "BAAI/bge-large-en-v1.5"
    BGE_M3 = "BAAI/bge-m3"


def get_embedding_model(
    model_name: AvailableEmbeddingModels = AvailableEmbeddingModels.BGE_LARGE_EN, model_kwargs: dict | None = None
) -> HuggingFaceEmbeddings:
    if model_kwargs is None:
        model_kwargs = {"device": "cuda"}
    return HuggingFaceEmbeddings(model_name=model_name.value, model_kwargs=model_kwargs)
