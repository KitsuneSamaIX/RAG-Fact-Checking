"""Evidence retrieval
"""

import os

import pandas as pd

from langchain_community.document_loaders import WebBaseLoader, BSHTMLLoader
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ContextualCompressionRetriever

from config import config


def create_retriever_from_urls(urls: pd.Series) -> BaseRetriever:
    return create_retriever_from_vector_store(_create_vector_store_from_urls(urls))


def create_retriever_from_vector_store(vector_store: VectorStore) -> BaseRetriever:
    if config.USE_RERANKER:
        n_docs_to_return = 20
    else:
        n_docs_to_return = config.TRUNCATED_RANKING_RETRIEVER_RESULTS

    vs_retriever = vector_store.as_retriever(
        search_type=config.VECTOR_STORE_SEARCH_TYPE,
        search_kwargs={'k': n_docs_to_return}
    )

    if config.USE_RERANKER:
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=config.get_document_compressor(), base_retriever=vs_retriever
        )
        return compression_retriever

    return vs_retriever


def _create_vector_store_from_urls(urls: pd.Series) -> VectorStore:
    result_docs = []

    for url in urls:
        try:
            if config.USE_CACHED_URLS:
                # Load from HTML file
                # Note: when config.USE_CACHED_URLS is True then each 'url' is a UUID
                if config.VERBOSE:
                    print(f"Loading cached URL: {url}")
                html_loader = BSHTMLLoader(os.path.join(config.CACHED_URLS_PATH, url + '.html'))
                docs = html_loader.load()
            else:
                # Load from web
                if config.VERBOSE:
                    print(f"Loading URL: {url}")
                web_loader = WebBaseLoader(url)
                docs = web_loader.load()

            # Split
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=config.TEXT_SPLITTER_CHUNK_SIZE,
                chunk_overlap=config.TEXT_SPLITTER_CHUNK_OVERLAP
            )
            docs = text_splitter.split_documents(docs)

            # Add to result
            result_docs += docs

        except Exception as e:
            print(f"WARNING: an exception occurred while loading the URL {url}. {e}")

    if config.VERBOSE:
        print("Building the vector store...")

    # Create a vector store from all the docs
    vector_store = FAISS.from_documents(result_docs, config.get_embeddings())

    return vector_store
