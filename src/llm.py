"""LLM model setup
"""

from langchain_core.runnables.base import Runnable
from langchain_core.embeddings.embeddings import Embeddings

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from config import config


def get_llm() -> Runnable:
    match config.MODEL_SERVER, config.MODEL_NAME:
        case 'ollama', 'llama3':
            return Ollama(model="llama3")
        case _:
            raise NotImplementedError()


def get_embeddings() -> Embeddings:
    match config.MODEL_SERVER, config.MODEL_NAME:
        case 'ollama', 'llama3':
            return OllamaEmbeddings(model="llama3")
        case _:
            raise NotImplementedError()
