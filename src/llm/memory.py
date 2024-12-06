
import json
import os
import logging
from langchain.memory import ConversationBufferMemory
from langchain_core.memory import BaseMemory
from langchain_community.chat_message_histories import FileChatMessageHistory




def create_memory(session_id) -> FileChatMessageHistory:
    print(f"Creating new memory for session {session_id}")
    return FileChatMessageHistory(file_path=f"./{session_id}")


def get_or_create_memory(
    session_id: str, storage_path: str = "."
) -> FileChatMessageHistory:
    """
    Retrieves a FileChatMessageHistory object from storage, or creates a new one if it doesn't exist.

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
            memory = json.load(file_object)
    else:
        logging.info(f"Creating new memory for session {session_id}")
        memory = create_memory()
    return memory



def get_memory_path(session_id: str, storage_path: str ) -> str:
    return f"{storage_path}/{session_id}"

# def dump_memory(
#     session_id: str,
#     memory: BaseMemory,
#     storage: ,
#     storage_path: str,
# ) -> None:
#     """
#     Serializes and saves a ConversationBufferMemory object to a storage.

#     Args:
#         session_id (str): The unique identifier for the session.
#         memory (ConversationBufferMemory): The memory object to be serialized and saved.
#         storage (AbstractStorage): The storage abstraction to handle saving operation.
#         storage_path (str): The base path in the storage system where the memory file will be saved.
#         Defaults to a configuration-defined chatlogs folder.

#     Returns:
#         None
#     """
#     memory_path = get_memory_path(session_id, storage_path)
#     with open(memory_path, "wb") as file_object:
#         pickle.dump(memory, file_object)



