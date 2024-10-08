# RAG Fact Checking
Fact Checking with Retrieval Augmented Generation (RAG) built with LangChain and fully configurable.

## Project
The aim of the project is the evaluation of different **LLMs** (Large Language Models) on fact checking tasks exploiting the RAG technique with different configuration parameters. It is designed to fact-check a predefined set of claims against their ground truth value.

The evaluation is meant to assess the impact of the **truncated ranking** on several standard metrics (i.e. _Accuracy, Precision, Recall, F1, MSE, MAE, etc._).

The **truncated ranking** parameter is the number of documents (_evidence_) that we use as context for the fact checking of a claim. The evidence is truncated (cut off) in order of relevance (i.e. _we keep the N most relevant documents_).

## Folders
- **src** contains the Fact Checking application's source files.
- **notebooks** contains jupyter notebooks to generate visualizations, manipulate data, etc.
- **playgrounds** contains random tests and tutorials.

## Notes
- If you use _search engine_ as retrieval mode, and you want to load each webpage from the web (instead of using cache), you should set the USER_AGENT environment variable.
