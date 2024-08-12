"""Configuration file

Note: keep sensitive data (like API keys) in environment variables.
"""

from langchain_core.language_models import BaseLanguageModel, BaseLLM, BaseChatModel
from langchain_core.embeddings import Embeddings

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEmbeddings


class _Common:
    # #################################
    # CORE
    # #################################

    # LLM
    LLM_TEMPERATURE = 0
    LLM_MAX_TOKENS = 20

    # Classification levels
    #   2 -> use True/False.
    #   6 -> use all 6 levels of the TRUTH-O-METER labels of "politifact.com".
    CLASSIFICATION_LEVELS = 2

    # Retrieval mode
    #   'bing+vs'   -> retrieve N results from Bing, build a vector store from those results and then retrieve N
    #                   documents from the vector store.
    #   'vs'        -> use a single vector store with all the evidence.
    RETRIEVAL_MODE = 'bing+vs'

    # Truncated ranking
    TRUNCATED_RANKING_RESULTS = 5

    # Data
    USE_SAMPLE = False  # Sample N (=SAMPLE_SIZE) IDs and the corresponding observations
    SAMPLE_SIZE = None

    # #################################
    # MODELS
    # #################################

    @classmethod
    def get_llm(cls) -> BaseLanguageModel:
        """LLM getter.

        Some model examples for each server:

        ------------------------------------------------------------

        Server: ollama

        Model names:
            llama3
            llama3.1
            llama3.1:8b-instruct-fp16

        Instance:
            ChatOllama(
                model=XXX,
                temperature=XXX,
                num_predict=XXX
            )

        ------------------------------------------------------------

        Server: vllm

        Model names:
            meta-llama/Meta-Llama-3.1-8B-Instruct

        Instance:
            ChatOpenAI(
                model=XXX,
                openai_api_key='EMPTY',
                openai_api_base=XXX,
                temperature=XXX,
                max_tokens=XXX
            )
        """
        raise NotImplementedError("You need to set a model in the config.")

    @classmethod
    def get_embeddings(cls) -> Embeddings:
        """Embeddings getter.

        Some model examples for each server:

        ------------------------------------------------------------

        Server: ollama

        Model names:
            nomic-embed-text

        Instance:
            OllamaEmbeddings(model=XXX)

        ------------------------------------------------------------

        Server: huggingface

        Model names:
            sentence-transformers/all-mpnet-base-v2

        Instance:
            HuggingFaceEmbeddings(
                model_name=XXX,
                model_kwargs={'device': 0},
                encode_kwargs={'normalize_embeddings': True},
                cache_folder=XXX
            )
        """
        raise NotImplementedError("You need to set a model in the config.")

    # #################################
    # BACKEND & DATA
    # #################################

    # Evidence
    USE_CACHED_URLS = True  # Makes sense only when using the URLs retrieved from the search engine (ex. Bing)

    # Vector store
    VECTOR_STORE_BACKEND = 'FAISS'  # Note: currently this parameter has NO effect, it is here only for documentation purposes

    # Data locations
    HUGGING_FACE_CACHE = None
    AGGR_DATA_PATH = None  # Location of the aggregated dataset
    ALL_EVIDENCE_VECTOR_STORE_PATH = None  # Location of the vector store containing all evidence

    # #################################
    # CODE
    # #################################

    # Verbose
    VERBOSE = False

    # Debug
    DEBUG = False
    SHOW_CONTEXT_FOR_DEBUG = False
    SHOW_PROMPT_FOR_DEBUG = False

    def __init__(self):
        """Default constructor.

        Checks for common oversights in configuration parameters.
        """
        cls = type(self)

        if cls.AGGR_DATA_PATH is None:
            raise RuntimeError("Configuration parameter 'AGGR_DATA_PATH' must be set.")

        if cls.USE_SAMPLE:
            if cls.SAMPLE_SIZE is None or not cls.SAMPLE_SIZE > 0:
                raise RuntimeError("Configuration parameter 'SAMPLE_SIZE' must be set and >0.")

        if cls.RETRIEVAL_MODE == 'vs':
            if cls.ALL_EVIDENCE_VECTOR_STORE_PATH is None:
                raise RuntimeError("Configuration parameter 'ALL_EVIDENCE_VECTOR_STORE_PATH' must be set.")


class _Local(_Common):
    AGGR_DATA_PATH = '/Users/mattia/Desktop/Lab avanzato 1 - RAG/Data/politifact-bing-retrieval/bert_aggregator_df.csv'

    @classmethod
    def get_llm(cls) -> BaseLanguageModel:
        return ChatOllama(
            model='llama3.1',
            temperature=cls.LLM_TEMPERATURE,
            num_predict=cls.LLM_MAX_TOKENS
        )

    @classmethod
    def get_embeddings(cls) -> Embeddings:
        return OllamaEmbeddings(model='nomic-embed-text')


class _LocalDebug(_Local):
    VERBOSE = True
    DEBUG = True
    SHOW_CONTEXT_FOR_DEBUG = False
    SHOW_PROMPT_FOR_DEBUG = False
    TRUNCATED_RANKING_RESULTS = 4
    USE_SAMPLE = True
    SAMPLE_SIZE = 5


class _UniudMitel3Server(_Common):
    HUGGING_FACE_CACHE = '/mnt/dmif-nas/SMDC/HF-Cache'

    RETRIEVAL_MODE = 'vs'
    ALL_EVIDENCE_VECTOR_STORE_PATH = '/mnt/dmif-nas/SMDC/datasets/Misinfo-Truncated-Rankings-RAG/data/cikm2024_soprano/embeddings/512'

    AGGR_DATA_PATH = '/mnt/dmif-nas/SMDC/politifact-bing-retrieval/bert_aggregator_df.csv'

    # @classmethod
    # def get_llm(cls) -> BaseLanguageModel:  # vllm (it mimics the OpenAI API)
    #     return ChatOpenAI(
    #         model='meta-llama/Meta-Llama-3.1-8B-Instruct',
    #         openai_api_key='EMPTY',
    #         openai_api_base='http://localhost:8005/v1',
    #         temperature=cls.LLM_TEMPERATURE,
    #         max_tokens=cls.LLM_MAX_TOKENS
    #     )

    @classmethod
    def get_llm(cls) -> BaseLanguageModel:
        return ChatOllama(
            model='llama3.1:8b-instruct-fp16',
            temperature=cls.LLM_TEMPERATURE,
            num_predict=cls.LLM_MAX_TOKENS
        )

    @classmethod
    def get_embeddings(cls) -> Embeddings:
        if cls.RETRIEVAL_MODE == 'vs':
            return HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-mpnet-base-v2',
                model_kwargs={'device': 0},
                encode_kwargs={'normalize_embeddings': True},
                cache_folder=cls.HUGGING_FACE_CACHE,
            )
        else:
            return OllamaEmbeddings(model='nomic-embed-text')


class _UniudMitel3ServerDebug(_UniudMitel3Server):
    VERBOSE = True
    DEBUG = True
    SHOW_CONTEXT_FOR_DEBUG = False
    SHOW_PROMPT_FOR_DEBUG = False
    TRUNCATED_RANKING_RESULTS = 4
    USE_SAMPLE = True
    SAMPLE_SIZE = 100


# Set config class
config = _LocalDebug
# config = _UniudMitel3ServerDebug
