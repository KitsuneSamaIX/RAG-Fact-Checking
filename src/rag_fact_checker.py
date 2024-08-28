"""RAG Fact Checker.

The RAGFactChecker class handles the RAG chain.
"""

from typing import Self

import bs4
import pandas as pd
import re

from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from retrieval import create_retriever_from_urls, create_retriever_from_vector_store
from common import Fact
from prompts import get_fact_checking_prompt_template, get_retry_msg
from config import config


class RAGFactChecker:
    """RAG based fact-checker.
    """

    def __init__(self, retriever: BaseRetriever):
        self._retriever = retriever

    @classmethod
    def from_urls(cls, urls: pd.Series) -> Self:
        """Builds a new RAGFactChecker.
        """
        return cls(create_retriever_from_urls(urls))

    @classmethod
    def from_vector_store(cls, vector_store: VectorStore) -> Self:
        """Builds a new RAGFactChecker.
        """
        return cls(create_retriever_from_vector_store(vector_store))

    def check(self, fact: Fact) -> bool | int | None:
        """Checks the truthfulness of the fact using RAG.

        Note: returns None when the LLM response cannot be parsed.
        """

        # Context retrieval  # TODO: add 'date' field to query (can it be useful?)
        retrieval_chain = self._retriever | self._format_docs
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
                | config.get_llm()
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

        # Set good responses
        match config.CLASSIFICATION_LEVELS:
            case 2:
                good_responses = ["TRUE", "FALSE"]
            case 6:
                good_responses = range(0, 6)
            case _:
                raise ValueError()

        # Validate the response
        if response not in good_responses:
            if config.DEBUG:
                print(f"\nRECEIVED BAD RESPONSE:\n{response}\n")
            if config.VERBOSE:
                print("Bad response, retrying...")

            retry_prompt = get_fact_checking_prompt_template()
            retry_prompt.append(("ai", response))
            retry_prompt.append(("human", get_retry_msg()))

            rag_chain = (
                    retry_prompt
                    | config.get_llm()
                    | StrOutputParser()
            )

            response = rag_chain.invoke(input_data)

        if config.VERBOSE:
            print(f"\nThe response is:\n{response}\n")

        return self._response_parser(response)

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _response_parser(response: str) -> bool | int | None:
        match config.CLASSIFICATION_LEVELS:
            case 2:
                return RAGFactChecker._response_parser_for_2_classification_levels(response)
            case 6:
                return RAGFactChecker._response_parser_for_6_classification_levels(response)
            case _:
                raise ValueError()

    @staticmethod
    def _response_parser_for_2_classification_levels(response: str) -> bool | None:
        if re.match(r'.*TRUE.*', response, re.IGNORECASE):
            return True
        elif re.match(r'.*FALSE.*', response, re.IGNORECASE):
            return False
        else:
            return None

    @staticmethod
    def _response_parser_for_6_classification_levels(response: str) -> int | None:
        match = re.search(r'\d+', response)

        if match:
            res = int(match.group())
            if res in range(0, 6):
                return res

        return None
