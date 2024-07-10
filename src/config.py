"""Configuration file

Note: keep sensitive data (like API keys) in environment variables.
"""


class _Common:
    # Model
    MODEL_SERVER = 'ollama'
    MODEL_NAME = 'llama3'

    # Truncated ranking
    TRUNCATED_RANKING_RESULTS = 5

    # Data
    AGGR_DATA_PATH = None
    USE_SAMPLE = False  # Samples N (=SAMPLE_SIZE) IDs and the corresponding observations
    SAMPLE_SIZE = None

    # Verbose
    VERBOSE = False

    # Debug
    DEBUG = False
    SHOW_CONTEXT_FOR_DEBUG = False
    SHOW_PROMPT_FOR_DEBUG = False

    def __init__(self):
        """Default constructor.

        Checks for common oversights in configuration parameters.
        """
        cls = type(self)

        if cls.AGGR_DATA_PATH is None:
            raise RuntimeError("Configuration parameter 'AGGR_DATA_PATH' must be set.")

        if cls.USE_SAMPLE:
            if cls.SAMPLE_SIZE is None or not cls.SAMPLE_SIZE > 0:
                raise RuntimeError("Configuration parameter 'SAMPLE_SIZE' must be set and >0.")


class _Local(_Common):
    AGGR_DATA_PATH = '/Users/mattia/Desktop/Lab avanzato 1 - RAG/Data/politifact-bing-retrieval/bert_aggregator_df.csv'


class _LocalDebug(_Local):
    VERBOSE = True
    DEBUG = True
    SHOW_CONTEXT_FOR_DEBUG = False
    SHOW_PROMPT_FOR_DEBUG = False
    TRUNCATED_RANKING_RESULTS = 2
    USE_SAMPLE = True
    SAMPLE_SIZE = 1


# Set config class
config = _LocalDebug()
