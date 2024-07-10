"""RAG chain setup

Composes the RAG chain.
"""

import bs4
import pandas as pd

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.vectorstores import VectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from llm import get_llm, get_embeddings
from fact import Fact
from config import config

_llm = get_llm()

_embeddings = get_embeddings()

_prompt_template = ChatPromptTemplate.from_template("""\
You are an assistant for fact checking.
You have to check if the <fact> said by the <speaker> is true based on the provided <context>.

Your response MUST be either:
- "TRUE" if the fact is true;
- "FALSE" if the fact is false.

<speaker>
{speaker}
</speaker>

<fact>
{fact}
</fact>

<context>
{context}
</context>
""")


def fact_check(fact: Fact, context_urls: pd.Series) -> bool:
    # Use the vector store as a retriever
    vs_retriever = _create_context_from_urls(context_urls).as_retriever(search_type="similarity", search_kwargs={"k": 6})

    # Retrieval chain
    retrieval_chain = (
        vs_retriever | _format_docs
    )

    # Context retrieval
    context_retrieval_query = "\n".join([fact.speaker, fact.text])
    context = retrieval_chain.invoke(context_retrieval_query)

    if config.DEBUG:
        print(f"\nCONTEXT RETRIEVAL QUERY:\n{context_retrieval_query}\n")

    if config.SHOW_CONTEXT_FOR_DEBUG:
        print(f"\nCONTEXT:\n{context}\n")

    # Compose the chain
    rag_chain = (
        _prompt_template
        | _llm
        | StrOutputParser()
    )

    if config.SHOW_PROMPT_FORMAT_FOR_DEBUG:
        print("\n\nPROMPT FORMAT:\n")
        print(
            _prompt_template.invoke({
                "speaker": "======PLACEHOLDER======",
                "fact": "======PLACEHOLDER======",
                "context": "======PLACEHOLDER======"
            }).to_messages()
        )
        print("\n")

    if config.VERBOSE:
        print("Invoking the RAG chain...")

    # Invoke the chain
    response = rag_chain.invoke({
        "speaker": fact.speaker,
        "fact": fact.text,
        "context": context
    })

    if response not in ["TRUE", "FALSE"]:
        # TODO.....
        pass

    print(f"\nTHE RESPONSE IS:\n{response}\n")

    # TODO aggiungere nella catena un prompt di estrazione del risulato TRUE o FALSE sulla base del fact checking (in altre parole fai l'operazine in due fasi, prima chiedi alla LLM di fare un fact checking, poi chiedile di estrarre TRUE/FALSE dal fact check)
    # TODO return bool


def _format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def _create_context_from_urls(urls: pd.Series) -> VectorStore:
    """Creates context with the content of the passed URLs.
    """
    result_docs = []

    for url in urls:
        try:
            # TODO class_=("post-content", "post-title", "post-header") (????)
            # TODO WebBaseLoader può caricare tutti gli url in un colpo solo senza il for (???)
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
