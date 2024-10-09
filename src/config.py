"""Centralized configuration.

Note:
- keep sensitive data (like API keys) in environment variables.

IMPORTANT:
 - Do NOT cache any of these configuration parameters (ex. in global variables),
    the orchestrator must be able to change any parameter before each run of the test suite.
"""

import os

from langchain_core.language_models import BaseLanguageModel, BaseLLM, BaseChatModel
from langchain_core.embeddings import Embeddings
from langchain_core.documents import BaseDocumentCompressor

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI

from langchain_huggingface import HuggingFaceEmbeddings

from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class _Common:
    # #################################
    # CORE
    # #################################

    # LLM
    LLM_TEMPERATURE = 0.1
    LLM_MAX_TOKENS = 100

    # Classification levels
    #   2 -> use True/False.
    #   6 -> use all 6 levels of the TRUTH-O-METER labels of "politifact.com".
    CLASSIFICATION_LEVELS = 2

    # Retrieval mode
    #   'se+vs'     -> retrieve N results from a search engine (ex. Bing), build a vector store from those results and
    #                   then retrieve N documents from the vector store.
    #   'vs'        -> use a single vector store with all the evidence.
    RETRIEVAL_MODE = 'se+vs'

    # Retrieval
    VECTOR_STORE_SEARCH_TYPE = 'similarity'
    USE_RERANKER = False

    # Retrieved evidence (documents)
    FILL_EVIDENCE = False  # Fill evidence with irrelevant documents until we have FILL_EVIDENCE_UPPER_LIMIT documents
    FILL_EVIDENCE_UPPER_LIMIT = None
    INVERT_EVIDENCE = False  # Invert the order of the retrieved documents

    # Text splitter
    TEXT_SPLITTER_CHUNK_SIZE = 1000  # Ignored when using pre-built vector stores
    TEXT_SPLITTER_CHUNK_OVERLAP = 200  # Ignored when using pre-built vector stores

    # Truncated ranking
    TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS = 6  # Keep the N highest ranking docs when retrieving from search engine results.
    TRUNCATED_RANKING_RETRIEVER_RESULTS = 8  # Keep the N highest ranking docs when retrieving from retriever results.

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

    @classmethod
    def get_document_compressor(cls) -> BaseDocumentCompressor:
        """Document compressor getter.

        Note: this is used for re-ranking.
        """
        raise NotImplementedError("You need to set a model in the config.")

    # #################################
    # BACKEND & DATA
    # #################################

    # >>> DATASET FORMAT <<<
    #
    # There are 3 types of datasets:
    #
    #   - GROUND_TRUTH_DATASET (required)
    #       - columns: id, speaker, text, date, target
    #
    #   - SEARCH_ENGINE_RESULTS_DATASET (optional)
    #       - columns: id, rank, snippet, url, uuid
    #
    #   - ALL_EVIDENCE_VECTOR_STORE (optional) (not a dataset but a prebuilt vector store with all evidence)
    #
    # Note: one between SEARCH_ENGINE_RESULTS_DATASET and ALL_EVIDENCE_VECTOR_STORE is required!

    # Dataset locations
    GROUND_TRUTH_DATASET_PATH = None
    SEARCH_ENGINE_RESULTS_DATASET_PATH = None

    # URLs
    USE_CACHED_URLS = True  # Makes sense only when using the URLs retrieved from a search engine (ex. Bing)
    CACHED_URLS_PATH = None

    # Vector store
    VECTOR_STORE_BACKEND = 'FAISS'  # Note: currently this parameter has NO effect, it is here only for documentation purposes
    ALL_EVIDENCE_VECTOR_STORE_PATH = None  # Location of the vector store containing all evidence
    ALL_EVIDENCE_VECTOR_STORE_INDEX_NAME = None

    # Other data locations
    HUGGING_FACE_CACHE_PATH = None
    RESULTS_PATH = None  # Location where to output tests' results

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
        raise RuntimeError("The configuration class is not meant to be instantiated.")

    @classmethod
    def check(cls):
        """Checks for common oversights in configuration parameters.
        """
        supported_classification_levels = [2, 6]
        if cls.CLASSIFICATION_LEVELS not in supported_classification_levels:
            raise RuntimeError(f"Configuration parameter 'CLASSIFICATION_LEVELS' must be in {supported_classification_levels}.")

        supported_retrieval_modes = ['se+vs', 'vs']
        if cls.RETRIEVAL_MODE not in supported_retrieval_modes:
            raise RuntimeError(f"Configuration parameter 'RETRIEVAL_MODE' must be in {supported_retrieval_modes}.")

        if cls.GROUND_TRUTH_DATASET_PATH is None:
            raise RuntimeError("Configuration parameter 'GROUND_TRUTH_DATASET_PATH' must be set.")

        if cls.USE_SAMPLE:
            if cls.SAMPLE_SIZE is None or not cls.SAMPLE_SIZE > 0:
                raise RuntimeError("Configuration parameter 'SAMPLE_SIZE' must be set and >0.")

        if cls.RETRIEVAL_MODE == 'se+vs' and cls.SEARCH_ENGINE_RESULTS_DATASET_PATH is None:
            raise RuntimeError("Configuration parameter 'SEARCH_ENGINE_RESULTS_DATASET_PATH' must be set.")

        if cls.RETRIEVAL_MODE == 'vs' and cls.ALL_EVIDENCE_VECTOR_STORE_PATH is None:
            raise RuntimeError("Configuration parameter 'ALL_EVIDENCE_VECTOR_STORE_PATH' must be set.")

        if cls.RETRIEVAL_MODE == 'vs' and cls.ALL_EVIDENCE_VECTOR_STORE_INDEX_NAME is None:
            raise RuntimeError("Configuration parameter 'ALL_EVIDENCE_VECTOR_STORE_INDEX_NAME' must be set.")

        if cls.RETRIEVAL_MODE == 'se+vs' and cls.USE_CACHED_URLS and cls.CACHED_URLS_PATH is None:
            raise RuntimeError("Configuration parameter 'CACHED_URLS_PATH' must be set.")

        if cls.RESULTS_PATH is None:
            raise RuntimeError("Configuration parameter 'RESULTS_PATH' must be set.")

        if cls.FILL_EVIDENCE:
            if cls.FILL_EVIDENCE_UPPER_LIMIT is None or not cls.FILL_EVIDENCE_UPPER_LIMIT > 0:
                raise RuntimeError("Configuration parameter 'FILL_EVIDENCE_UPPER_LIMIT' must be set and >0.")

    @classmethod
    def get_printable_config(cls) -> str:
        """Returns a printable text of the key configuration parameters.
        """
        lines = []
        lines.append("\nDATASET:")
        lines.append(f" - GROUND_TRUTH_DATASET_PATH: {cls.GROUND_TRUTH_DATASET_PATH}")
        lines.append("\nCONFIGURATION:")
        lines.append(f" - LLM (model): {cls.get_llm()}")
        lines.append(f" - Embeddings (model): {cls.get_embeddings()}")
        lines.append(f" - LLM_TEMPERATURE: {cls.LLM_TEMPERATURE}")
        lines.append(f" - LLM_MAX_TOKENS: {cls.LLM_MAX_TOKENS}")
        lines.append(f" - CLASSIFICATION_LEVELS: {cls.CLASSIFICATION_LEVELS}")
        lines.append(f" - RETRIEVAL_MODE: {cls.RETRIEVAL_MODE}")
        lines.append(f" - VECTOR_STORE_SEARCH_TYPE: {cls.VECTOR_STORE_SEARCH_TYPE}")
        lines.append(f" - USE_RERANKER: {cls.USE_RERANKER}")
        if cls.USE_RERANKER:
            lines.append(f" - Cross-Encoder (model): {cls.get_document_compressor()}")
        lines.append(f" - FILL_EVIDENCE: {cls.FILL_EVIDENCE}")
        if cls.FILL_EVIDENCE:
            lines.append(f" - FILL_EVIDENCE_UPPER_LIMIT: {cls.FILL_EVIDENCE_UPPER_LIMIT}")
        lines.append(f" - INVERT_EVIDENCE: {cls.INVERT_EVIDENCE}")
        if cls.RETRIEVAL_MODE == 'se+vs':
            lines.append(f" - TEXT_SPLITTER_CHUNK_SIZE: {cls.TEXT_SPLITTER_CHUNK_SIZE}")
            lines.append(f" - TEXT_SPLITTER_CHUNK_OVERLAP: {cls.TEXT_SPLITTER_CHUNK_OVERLAP}")
            lines.append(f" - TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS: {cls.TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS}")
        lines.append(f" - TRUNCATED_RANKING_RETRIEVER_RESULTS: {cls.TRUNCATED_RANKING_RETRIEVER_RESULTS}")
        text = '\n'.join(lines)
        return text

    @classmethod
    def print_config(cls):
        """Prints the key configuration parameters.
        """
        print(cls.get_printable_config())


class _Local(_Common):
    DATASET_PATH_PREFIX = '/Users/mattia/Desktop/Lab avanzato 1 - RAG/Data/'
    DATASET_NAME = 'cikm2024_soprano_clean'
    DATASET_NAME_2 = 'cikm2024_soprano'

    GROUND_TRUTH_DATASET_PATH = os.path.join(DATASET_PATH_PREFIX, DATASET_NAME, 'ground_truth.csv')
    SEARCH_ENGINE_RESULTS_DATASET_PATH = os.path.join(DATASET_PATH_PREFIX, DATASET_NAME_2, 'df_evidence_list-top10.csv')
    ALL_EVIDENCE_VECTOR_STORE_PATH = os.path.join(DATASET_PATH_PREFIX, DATASET_NAME, 'embeddings/faiss_nomic_embed_text_chunk_size_1000/')
    ALL_EVIDENCE_VECTOR_STORE_INDEX_NAME = 'index'
    CACHED_URLS_PATH = os.path.join(DATASET_PATH_PREFIX, DATASET_NAME_2, 'evidence_to_index/')
    RESULTS_PATH = '/Users/mattia/Desktop/Lab avanzato 1 - RAG/Results/'

    @classmethod
    def get_llm(cls) -> BaseLanguageModel:
        return ChatOllama(
            model='llama3.2',
            temperature=cls.LLM_TEMPERATURE,
            num_predict=cls.LLM_MAX_TOKENS
        )

    @classmethod
    def get_embeddings(cls) -> Embeddings:
        return OllamaEmbeddings(model='nomic-embed-text')

    @classmethod
    def get_document_compressor(cls) -> BaseDocumentCompressor:
        model = HuggingFaceCrossEncoder(
            model_name='BAAI/bge-reranker-base'
        )
        compressor = CrossEncoderReranker(model=model, top_n=cls.TRUNCATED_RANKING_RETRIEVER_RESULTS)
        return compressor


class _LocalDebug(_Local):
    VERBOSE = True
    DEBUG = True
    SHOW_CONTEXT_FOR_DEBUG = True
    # SHOW_PROMPT_FOR_DEBUG = True
    USE_SAMPLE = True
    SAMPLE_SIZE = 1
    RETRIEVAL_MODE = 'vs'
    # USE_RERANKER = True
    CLASSIFICATION_LEVELS = 2
    FILL_EVIDENCE = True
    FILL_EVIDENCE_UPPER_LIMIT = 10
    INVERT_EVIDENCE = True


class _UniudMitel3Server(_Common):
    DATASET_PATH_PREFIX = '/mnt/dmif-nas/SMDC/datasets/Misinfo-Truncated-Rankings-RAG/data/'
    DATASET_NAME = 'cikm2024_soprano'

    GROUND_TRUTH_DATASET_PATH = os.path.join(DATASET_PATH_PREFIX, DATASET_NAME, 'ground_truth.csv')
    # SEARCH_ENGINE_RESULTS_DATASET_PATH = os.path.join(DATASET_PATH_PREFIX, DATASET_NAME, 'df_evidence_list-top10.csv')
    ALL_EVIDENCE_VECTOR_STORE_PATH = os.path.join(DATASET_PATH_PREFIX, DATASET_NAME, 'embeddings/512/')
    ALL_EVIDENCE_VECTOR_STORE_INDEX_NAME = 'FAISS_INDEX_CHUNK_SIZE_512'
    # CACHED_URLS_PATH = os.path.join(DATASET_PATH_PREFIX, DATASET_NAME, 'evidence_to_index/')
    HUGGING_FACE_CACHE_PATH = '/mnt/dmif-nas/SMDC/HF-Cache'
    RESULTS_PATH = '/home/fedrigo/results/RAG-Fact-Checking/'

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
            # model='llama3.1:8b-instruct-fp16',
            model='mistral-nemo:12b-instruct-2407-fp16',
            temperature=cls.LLM_TEMPERATURE,
            num_predict=cls.LLM_MAX_TOKENS
        )

    @classmethod
    def get_embeddings(cls) -> Embeddings:
        if cls.RETRIEVAL_MODE == 'vs':
            return HuggingFaceEmbeddings(
                model_name='sentence-transformers/all-mpnet-base-v2',
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True},
                cache_folder=cls.HUGGING_FACE_CACHE_PATH
            )
        else:
            return OllamaEmbeddings(model='nomic-embed-text')

    @classmethod
    def get_document_compressor(cls) -> BaseDocumentCompressor:
        model = HuggingFaceCrossEncoder(
            model_name='BAAI/bge-reranker-base'
        )
        compressor = CrossEncoderReranker(model=model, top_n=cls.TRUNCATED_RANKING_RETRIEVER_RESULTS)
        return compressor


class _UniudMitel3ServerDebug(_UniudMitel3Server):
    # VERBOSE = True
    # DEBUG = True
    # SHOW_CONTEXT_FOR_DEBUG = True
    # SHOW_PROMPT_FOR_DEBUG = True
    # USE_SAMPLE = True
    SAMPLE_SIZE = 100
    RETRIEVAL_MODE = 'vs'
    # USE_RERANKER = True
    CLASSIFICATION_LEVELS = 2


# Set config class
# config = _LocalDebug
config = _UniudMitel3ServerDebug

config.check()
