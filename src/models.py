"""Models setup
"""

from langchain_core.language_models import BaseLanguageModel, BaseLLM, BaseChatModel
from langchain_core.embeddings import Embeddings

from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import config


def get_llm() -> BaseLanguageModel:
    match config.LLM_SERVER, config.LLM_NAME:
        # Server: ollama
        case 'ollama', name:
            return ChatOllama(model=name)  # TODO try setting the temperature (the actual default is 0.8, I think) and also other parameters of ChatOllama
        # Some model names:
        #   llama3
        #   llama3.1
        #   llama3.1:8b-instruct-fp16

        # Server: vllm
        case 'vllm', name:
            return ChatOpenAI(
                model=name,
                openai_api_key='EMPTY',
                openai_api_base=config.LLM_SERVER_URL,
                max_tokens=5,
                temperature=0
            )
        # Some model names:
        #   meta-llama/Meta-Llama-3.1-8B-Instruct

        # Default
        case _:
            raise NotImplementedError()


def get_embeddings() -> Embeddings:
    match config.EMBEDDINGS_SERVER, config.EMBEDDINGS_NAME:
        # Server: ollama
        case 'ollama', name:
            return OllamaEmbeddings(model=name)
        # Some model names:
        #   nomic-embed-text

        # Server: vllm
        case 'vllm', name:
            return OpenAIEmbeddings(
                model=name,
                openai_api_key='EMPTY',
                openai_api_base=config.EMBEDDINGS_SERVER_URL
            )
        # Some model names:
        #   nomic-ai/nomic-embed-text-v1.5 (doesn't work with vllm)

        # Default
        case _:
            raise NotImplementedError()
