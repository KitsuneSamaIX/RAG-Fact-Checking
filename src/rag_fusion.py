"""RAG Fusion

LangChain implementation of https://github.com/Raudaschl/rag-fusion.
"""

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.retrievers import BaseRetriever
from langchain.load import dumps, loads

from config import config


def rag_fusion(original_query: str, retriever: BaseRetriever) -> list[Document]:
    """Retrieve and rerank results using the RAG Fusion technique.
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that generates multiple search queries based on a single input query."),
        ("human", "Generate multiple search queries related to: {original_query}"),
        ("human", "OUTPUT (4 queries):")
    ])

    generate_queries = (
        prompt | config.get_llm_for_rag_fusion() | StrOutputParser() | (lambda x: x.split("\n"))
    )

    chain = generate_queries | retriever.map() | _reciprocal_rank_fusion

    docs = chain.invoke({'original_query': original_query})

    return docs


def _reciprocal_rank_fusion(results: list[list], k=60) -> list[Document]:
    fused_scores = {}

    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    # reranked_results = [
    #     (loads(doc), score)
    #     for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    # ]

    reranked_results = [
        loads(doc)
        for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]

    return reranked_results
