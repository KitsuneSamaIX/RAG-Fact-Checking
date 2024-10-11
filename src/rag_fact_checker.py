"""RAG Fact Checker.

The RAGFactChecker class handles the RAG chain.
"""

from typing import Self

import pandas as pd
import re
import random

from langchain_core.output_parsers import StrOutputParser
from langchain_core.vectorstores import VectorStore
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import BaseMessage

from retrieval import create_vector_store_from_urls, create_retriever_from_vector_store
from common import Fact
from prompts import get_fact_checking_prompt_template, get_retry_msg
from config import config


class RAGFactChecker:
    """RAG based fact-checker.
    """

    def __init__(self, retriever: BaseRetriever, vector_store: VectorStore):
        """Inits the RAGFactChecker.

        - The 'retriever' is used for the RAG.
        - The raw 'vector_store' is used for the (optional) evidence fill.
        """
        self._retriever = retriever
        self._vector_store = vector_store

    @classmethod
    def from_urls(cls, urls: pd.Series) -> Self:
        """Builds a new RAGFactChecker.
        """
        vector_store = create_vector_store_from_urls(urls)
        return cls(create_retriever_from_vector_store(vector_store), vector_store)

    @classmethod
    def from_vector_store(cls, vector_store: VectorStore) -> Self:
        """Builds a new RAGFactChecker.
        """
        return cls(create_retriever_from_vector_store(vector_store), vector_store)

    def check(self, fact: Fact) -> int | None:
        """Checks the truthfulness of the fact using RAG.

        Note: returns None when the LLM response cannot be parsed.
        """
        # Statement to check
        if pd.isna(fact.speaker):
            statement = fact.text
        else:
            statement = fact.speaker + " said " + fact.text

        # Context retrieval
        retrieval_chain = self._retriever | self._fill_docs | self._invert_docs | self._format_docs
        context_retrieval_query = statement
        context = retrieval_chain.invoke(context_retrieval_query)

        # Input data
        input_data = {
            'statement': statement,
            'context': context
        }

        # Main chain
        self._check_exceeded_context_length(get_fact_checking_prompt_template().invoke(input_data).to_messages())
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
                good_responses = ['TRUE', 'FALSE', 'True', 'False', 'true', 'false']
            case 6:
                good_responses = range(0, 6)
            case _:
                raise ValueError()

        # Validate the response and retry one time if necessary
        if response not in good_responses:
            if config.DEBUG:
                print(f"\nRECEIVED BAD RESPONSE:\n{response}\n")
            if config.VERBOSE:
                print("Bad response, retrying...")

            retry_prompt = get_fact_checking_prompt_template()
            retry_prompt.append(("ai", response))
            retry_prompt.append(("human", get_retry_msg()))

            self._check_exceeded_context_length(retry_prompt.invoke(input_data).to_messages())
            rag_chain = (
                retry_prompt
                | config.get_llm()
                | StrOutputParser()
            )

            response = rag_chain.invoke(input_data)

        if config.VERBOSE:
            print(f"\nThe response is: {response}\n")

        return self._response_parser(response)

    @staticmethod
    def _format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    @staticmethod
    def _invert_docs(docs: list[Document]) -> list[Document]:
        if config.INVERT_EVIDENCE:
            docs.reverse()
        return docs

    def _fill_docs(self, docs: list[Document]) -> list[Document]:
        """Fills evidence with irrelevant documents until we have FILL_EVIDENCE_UPPER_LIMIT documents.
        """
        if config.FILL_EVIDENCE:
            n_docs_to_add = config.FILL_EVIDENCE_UPPER_LIMIT - len(docs)
            if n_docs_to_add > 0:
                embedding = config.get_embeddings().embed_query("")  # Embed an empty query to get the vector's format
                for i in range(0, len(embedding)):
                    embedding[i] = random.uniform(-0.2, 0.2)  # Randomize the vector
                docs_to_add = self._vector_store.similarity_search_by_vector(embedding, k=n_docs_to_add)
                docs = docs + docs_to_add
        return docs

    @staticmethod
    def _response_parser(response: str) -> int | None:
        match config.CLASSIFICATION_LEVELS:
            case 2:
                return RAGFactChecker._response_parser_for_2_classification_levels(response)
            case 6:
                return RAGFactChecker._response_parser_for_6_classification_levels(response)
            case _:
                raise ValueError()

    @staticmethod
    def _response_parser_for_2_classification_levels(response: str) -> int | None:
        if re.match(r'.*TRUE.*', response, re.IGNORECASE):
            return 1
        elif re.match(r'.*FALSE.*', response, re.IGNORECASE):
            return 0
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

    @staticmethod
    def _check_exceeded_context_length(messages: list[BaseMessage]):
        """Checks if the number of tokens needed to encode all messages exceeds the model's context length.

        Raises a RuntimeError if it exceeds.
        """
        n_tokens = config.get_llm().get_num_tokens_from_messages(messages)
        if n_tokens > config.LLM_CONTEXT_LENGTH:
            raise RuntimeError(f"The number of tokens needed to encode all messages ({n_tokens}) exceeds the model's context length ({config.LLM_CONTEXT_LENGTH}).")
