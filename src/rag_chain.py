"""RAG chain setup

Composes the RAG chain.
"""

import bs4
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter

from llm import get_llm, get_embeddings
from fact import Fact

llm = get_llm()
embeddings = get_embeddings()


# TODO: WORK IN PROGRESS...........


def fact_check(fact: Fact, context_urls: pd.Series) -> bool:
    # TODO: build the prompt
    pass


def _create_context_from_urls(urls: pd.Series) -> FAISS:
    """Creates context with the content of the passed URLs.
    """
    result_docs = []

    for url in urls:
        # TODO: add error handling?? (try-except)
        # Load from web
        web_loader = WebBaseLoader(url)
        docs = web_loader.load()

        # Split
        text_splitter = RecursiveCharacterTextSplitter()
        docs = text_splitter.split_documents(docs)

        # Add to result
        result_docs += docs

    # Create a vector store from all the docs
    vector_store = FAISS.from_documents(result_docs, embeddings)

    return vector_store


# TODO: reference code below (check if you need some of those options)

# Load, chunk and index the contents of the blog.
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# rag_chain = (
#     {"context": retriever | format_docs, "question": RunnablePassthrough()}
#     | prompt
#     | llm
#     | StrOutputParser()
# )

# rag_chain.invoke("What is Task Decomposition?")
