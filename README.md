# Local DnD LLM Assistant

This application is a local LLM powered Dungeon Master Assistant Chat powered by ollama. The idea is to help Dungeon Masters (mainly me) manage their games by providing planning assistance, note management, and information retrieval.

## Features

- Conversational interface for Dungeon Masters
- Model management and selection
- Document and memory retrieval
- Chat logging with MLFlow

## Usage

1. Set up the environment and env variables (explained below).
2. (Optional) Additional setup steps.
3. Set up Ollama.
4. Run the application:

```sh
./start_app.sh
```

### Installation
Create and activate a virtual environment:

```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```

Install the required dependencies
```bash
pip install -r requirements.txt
```

(Optionally) Set up DuckDB with your configuration. The app should set up a working version once you upload documents.

Run the application:
```bash
./start_app.sh
```

### Configuration
The application configuration can be found in the config module. Key configuration options include:

`default_chatlog_path`: Path to save chat logs.
`app_chat_model`: Default chat model to use.
`app_chat_temperature`: Temperature setting for the chat model.
`app_chat_context_length`: Context length for the chat model.
`app_embedding_model`: Default embedding model to use.


### Open TODOs:

- [ ] make documents filterable (e.g. RAG only on specific docs)
- [ ] introduce single document mode (read in whole doc instead of RAG)
- [ ] test other models
- [ ] allow for selection of previous chat memory files
- [ ] improve PDF parsing and chunking
- [ ] automate llm eval outside of app
- [ ] add different chat modes (e.g. story telling, question answering etc.)
- [ ] refactor some features
