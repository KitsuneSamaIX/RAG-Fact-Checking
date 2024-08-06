"""RAG chain setup

Composes the RAG chain.
"""

import bs4
import pandas as pd
import re

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from models import get_llm, get_embeddings
from fact import Fact
from prompts import get_fact_checking_prompt_template, retry_msg
from config import config

_llm = get_llm()

_embeddings = get_embeddings()


def fact_check(fact: Fact, context_urls: pd.Series) -> bool | None:
    # Use the vector store as a retriever
    vs_retriever = _create_context_from_urls(context_urls).as_retriever(search_type="similarity", search_kwargs={"k": 8})

    # Retrieval chain
    retrieval_chain = (
        vs_retriever | _format_docs
    )

    # Context retrieval
    context_retrieval_query = "\n".join([fact.speaker, fact.text])
    context = retrieval_chain.invoke(context_retrieval_query)

    # Input data
    input_data = {
        "speaker": fact.speaker,
        "fact": fact.text,
        "context": context
    }

    # Main chain
    rag_chain = (
            get_fact_checking_prompt_template()
            | _llm
            | StrOutputParser()
    )

    if config.DEBUG:
        print(f"\nCONTEXT RETRIEVAL QUERY:\n{context_retrieval_query}\n")

    if config.SHOW_CONTEXT_FOR_DEBUG:
        print(f"\nCONTEXT:\n{context}\n")

    if config.SHOW_PROMPT_FOR_DEBUG:
        print(f"FORMATTED PROMPT:\n{get_fact_checking_prompt_template().invoke(input_data)}\n")

    if config.VERBOSE:
        print("Invoking the RAG chain...")

    # Invoke the chain
    response = rag_chain.invoke(input_data)

    # Validate the response
    if response not in ["TRUE", "FALSE"]:
        if config.DEBUG:
            print(f"\nRECEIVED NON BINARY RESPONSE:\n{response}\n")
        if config.VERBOSE:
            print("Non binary response, retrying...")

        retry_prompt = get_fact_checking_prompt_template()
        retry_prompt.append(("ai", response))
        retry_prompt.append(("human", retry_msg))

        rag_chain = (
            retry_prompt
            | _llm
            | StrOutputParser()
        )

        response = rag_chain.invoke(input_data)

    if config.VERBOSE:
        print(f"\nThe response is:\n{response}\n")

    return _response_parser(response)


def _response_parser(response: str) -> bool | None:
    if re.match(r".*TRUE.*", response, re.IGNORECASE):
        return True
    elif re.match(r".*FALSE.*", response, re.IGNORECASE):
        return False
    else:
        return None


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _create_context_from_urls(urls: pd.Series) -> VectorStore:
    """Creates context with the content of the passed URLs.
    """
    result_docs = []

    for url in urls:
        try:
            if config.VERBOSE:
                print(f"Loading URL: {url}")

            # Load from web
            web_loader = WebBaseLoader(url)
            docs = web_loader.load()

            # Split
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.split_documents(docs)

            # Add to result
            result_docs += docs

        except Exception as e:
            print(f"WARNING: an exception occurred while loading the URL {url}. {e}")

    if config.VERBOSE:
        print("Building the vector store...")

    # Create a vector store from all the docs
    vector_store = FAISS.from_documents(result_docs, _embeddings)

    return vector_store
