from pathlib import Path

import dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from src.llm.deployments import AvailableChatModels, AvailableEmbeddingModels


class Settings(BaseSettings):
    """Settings with prioritized loading from environment variables"""

    model_config = SettingsConfigDict(
        env_file=f"{Path(__file__).parent}/.env",
        env_file_encoding="utf-8",
        # Allow population of fields not explicitly defined in the model
        # necessary for dynamic creation of openai config
        extra="ignore",
    )
    # could potentially be moved to Configuration with BaseSettings as Parent


class Configuration(Settings):
    app_chat_model: AvailableChatModels = Field(
        default=AvailableChatModels.LLAMA_3_2, validation_alias="APP_CHAT_MODEL"
    )
    app_embedding_model: AvailableEmbeddingModels = Field(
        default=AvailableEmbeddingModels.BGE_LARGE_EN, validation_alias="APP_EMBEDDING_MODEL"
    )

    memory_storage_path: str = Field(default="./src/chatlogs", validation_alias="MEMORY_STORAGE_PATH")
    app_chat_temperature: float = Field(default=0.5, validation_alias="APP_CHAT_TEMPERATURE")
    app_chat_context_length: int = Field(default=32000, validation_alias="APP_CHAT_CONTEXT_LENGTH")

    default_vector_key: str = Field(default="embedding", validation_alias="DEFAULT_VECTOR_KEY")
    default_id_key: str = Field(default="id", validation_alias="DEFAULT_ID_KEY")
    default_text_key: str = Field(default="text", validation_alias="DEFAULT_TEXT_KEY")
    default_table_name: str = Field(default="embeddings", validation_alias="DEFAULT_TABLE_NAME")
    default_document_id_column: str = Field(default="reference", validation_alias="DEFAULT_DOCUMENT_ID_COLUMN")
    similarity_alias: str = Field(default="similarity_score", validation_alias="SIMILARITY_ALIAS")

    default_chunk_size: int = Field(default=4000, validation_alias="DEFAULT_CHUNK_SIZE")
    default_chunk_overlap: int = Field(default=100, validation_alias="DEFAULT_CHUNK_OVERLAP")


dotenv.load_dotenv(dotenv.find_dotenv(), override=True)
config = Configuration()
