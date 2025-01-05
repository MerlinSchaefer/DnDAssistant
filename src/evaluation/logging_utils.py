import os
from datetime import datetime

import mlflow

from src.config.configuration import config

DEFAULT_STORAGE_PATH = config.default_chatlog_path


def log_chat(message: str, response: str, log_file: str | None = None) -> None:
    """
    Logs input message and response to MLFlow as an artifact containing a string
    of the message and response.

    Args:
        message (str): The user's input message.
        response (str): The assistant's response.
        log_file (str): The file path to save the chat log.
    Returns:
        None
    """
    if log_file is None:
        log_file = f"{DEFAULT_STORAGE_PATH}/mlflow_logging/chatlog_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt"

    # Check if the log file already exists
    if os.path.exists(log_file):
        print(f"Appending to existing log file: {log_file}")
        with open(log_file, "a") as f:
            f.write(f"USER:{message}\nLLM:{response}\n")
    else:
        with open(log_file, "w") as f:
            f.write(f"USER:{message}\nLLM:{response}\n")

    mlflow.log_artifact(log_file)
