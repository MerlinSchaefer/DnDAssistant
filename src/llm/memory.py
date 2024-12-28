import logging
import os
import pickle

from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_core.memory import BaseMemory

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


# TODO: handle loading and saving in agent
def create_buffer_memory() -> ConversationBufferMemory:
    return ConversationBufferMemory(memory_key="chat_history", return_messages=True)


def dump_memory(
    session_id: str,
    memory: BaseMemory,
    storage_path: str = DEFAULT_STORAGE_PATH,
) -> None:
    """
    Serializes and saves a ConversationBufferMemory object to a storage.

    Args:
        session_id (str): The unique identifier for the session.
        memory (ConversationBufferMemory): The memory object to be serialized and saved.
        storage_path (str): The base path in the storage system where the memory file will be saved.
        Defaults to a configuration-defined chatlogs folder.

    Returns:
        None
    """
    # TODO: Handle case where memory is None.
    memory_path = get_memory_path(session_id, storage_path)
    with open(memory_path, "wb") as file_object:
        pickle.dump(memory, file_object)


def get_or_create_buffer_memory(session_id: str, storage_path: str) -> ConversationBufferMemory:
    """
    Retrieves a ConversationBufferMemory object from storage, or creates a new one if it doesn't exist.

    Args:
        session_id: The unique identifier for the session.
        storage_path: The base path in the storage system where the memory file would be located.
            Defaults to a configuration-defined chatlogs folder.

    Returns:
        ConversationBufferMemory: The loaded or newly created memory object.
    """
    memory_path = get_memory_path(session_id, storage_path)
    # Check if the memory exists
    if os.path.exists(memory_path):
        logging.info(f"Loading memory for session {session_id}")
        # TODO storage path make project-chatlog path
        with open(memory_path, "rb") as file_object:
            memory = pickle.load(file_object)
    else:
        logging.info(f"Creating new memory for session {session_id}")
        memory = create_buffer_memory()
    return memory


class PickleMemory:  # TODO rename to something like MemoryHandler
    """Chat memory that is stored in a pickle file."""

    def __init__(self, session_id: str, storage_path: str = DEFAULT_STORAGE_PATH):
        if not storage_path:
            self.storage_path = DEFAULT_STORAGE_PATH
        else:
            self.storage_path = storage_path
        self.session_id = session_id
        self.memory = create_buffer_memory()

    def save(self):
        dump_memory(self.session_id, self.memory, self.storage_path)  # type: ignore

    def load(self):
        self.memory = get_or_create_buffer_memory(session_id=self.session_id, storage_path=self.storage_path)

    def get_memory(self) -> ConversationBufferMemory:
        return self.memory
