"""Main module.

This is the entrypoint of the app.
"""

import pandas as pd
from orchestration import run_orchestrator, run_test


if __name__ == '__main__':
    pd.set_option('mode.copy_on_write', True)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_rows', 5)

    # run_test()
    run_orchestrator()


# TODO use rerank across different embeddings sizes (as seen on https://aman.ai/primers/ai/RAG/#figuring-out-the-ideal-chunk-size)
