"""Models setup
"""

from langchain_core.language_models import BaseLanguageModel, BaseLLM, BaseChatModel
from langchain_core.embeddings import Embeddings

from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama, ChatOpenAI
from langchain_community.embeddings import OllamaEmbeddings, OpenAIEmbeddings  # TODO: replace imports subject to future deprecation

from config import config


def get_llm() -> BaseLanguageModel:
    match config.LLM_SERVER, config.LLM_NAME:
        case 'ollama', 'llama3':
            return ChatOllama(model="llama3")
        case 'ollama', 'llama3.1':
            return ChatOllama(model="llama3.1")
        case 'vllm', 'llama3.1':
            return ChatOpenAI(
                model='meta-llama/Meta-Llama-3.1-8B-Instruct',
                openai_api_key='EMPTY',
                openai_api_base=config.LLM_SERVER,
                max_tokens=5,
                temperature=0
            )
        case _:
            raise NotImplementedError()


def get_embeddings() -> Embeddings:
    match config.EMBEDDINGS_SERVER, config.EMBEDDINGS_NAME:
        case 'ollama', 'llama3':
            return OllamaEmbeddings(model="llama3")
        case 'ollama', 'nomic-embed-text':
            return OllamaEmbeddings(model="nomic-embed-text")
        case 'vllm', 'nomic-embed-text':
            return OpenAIEmbeddings(
                model='nomic-ai/nomic-embed-text-v1.5',
                openai_api_key='EMPTY',
                openai_api_base=config.LLM_SERVER
            )
        case _:
            raise NotImplementedError()
