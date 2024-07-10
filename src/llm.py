"""Models setup
"""

from langchain_core.language_models import BaseLLM
from langchain_core.embeddings import Embeddings

from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

from config import config


def get_llm() -> BaseLLM:
    match config.LLM_SERVER, config.LLM_NAME:
        case 'ollama', 'llama3':
            return Ollama(model="llama3")
        case _:
            raise NotImplementedError()


def get_embeddings() -> Embeddings:
    match config.EMBEDDINGS_SERVER, config.EMBEDDINGS_NAME:
        case 'ollama', 'llama3':
            return OllamaEmbeddings(model="llama3")
        case 'ollama', 'nomic-embed-text':
            return OllamaEmbeddings(model="nomic-embed-text")
        case _:
            raise NotImplementedError()
