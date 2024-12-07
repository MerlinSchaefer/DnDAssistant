import os

from langchain_community.chat_message_histories import FileChatMessageHistory

DEFAULT_STORAGE_PATH = "./src/chatlogs"


def get_or_create_memory(session_id: str, storage_path: str = DEFAULT_STORAGE_PATH) -> FileChatMessageHistory:
    """
    Retrieves a FileChatMessageHistory object from storage, or creates a new one if it doesn't exist.

    Args:
        session_id: The unique identifier for the session.
        storage_path: The base path in the storage system where the memory file would be located.
            Defaults to a configuration-defined chatlogs folder.

    Returns:
        ConversationBufferMemory: The loaded or newly created memory object.
    """
    # Ensure the storage directory exists
    if not os.path.exists(storage_path):
        os.makedirs(storage_path)
    memory_path = get_memory_path(session_id, storage_path)

    return FileChatMessageHistory(file_path=memory_path)


def list_memories(storage_path: str = DEFAULT_STORAGE_PATH) -> list[str]:
    """
    List all the memory files stored in the storage system.

    Args:
        storage_path: The base path in the storage system where the memory files are located.
            Defaults to a configuration-defined chatlogs folder.

    Returns:
        list[str]: A list of paths to the memory files.
    """
    if not os.path.exists(storage_path):
        return []
    return [f"{storage_path}/{f}" for f in os.listdir(storage_path) if os.path.isfile(f"{storage_path}/{f}")]


def create_memory(memory_path) -> FileChatMessageHistory:
    print(f"Creating new memory at {memory_path}")
    return FileChatMessageHistory(file_path=memory_path)


def get_memory_path(session_id: str, storage_path: str) -> str:
    return f"{storage_path}/{session_id}"
