import json
import logging
import os

from langchain_community.chat_message_histories import FileChatMessageHistory


def get_or_create_memory(session_id: str, storage_path: str = "./chatlogs") -> FileChatMessageHistory:
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
    # Check if the memory exists
    if os.path.exists(memory_path):
        logging.info(f"Loading memory for session {session_id}")
        with open(memory_path, "rb") as file_object:
            memory = json.load(file_object)
    else:
        logging.info(f"Creating new memory for session {session_id}")
        memory = create_memory(memory_path)
    return memory


def create_memory(memory_path) -> FileChatMessageHistory:
    print(f"Creating new memory at {memory_path}")
    return FileChatMessageHistory(file_path=memory_path)


def get_memory_path(session_id: str, storage_path: str) -> str:
    return f"{storage_path}/{session_id}"
