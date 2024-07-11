# RAG Fact Checking
Fact Checking with Retrieval Augmented Generation (RAG) built with LangChain and fully configurable.

## Notes
- This is a test version for performance evaluation, it is designed to fact-check a predefined set of claims against their ground truth value.
    - The retrieval of information to fact-check the claim uses a predefined set of static (scraped) search data and provides a ranked list of URLs for each (predefined) claim. 
      - The content of the retrieved URLs is loaded each time from the web.
      - It is easy to swap the static search with a real search engine (ex. Google, Bing, etc.) but it requires the use of paid APIs.
- You should set the USER_AGENT environment variable before usage.
