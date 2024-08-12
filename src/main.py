"""Main

This is the entrypoint of my RAG for Fake News App.
"""

import time
import pandas as pd
from test_suite import run_test_suite

if __name__ == '__main__':
    pd.set_option('mode.copy_on_write', True)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_rows', 5)

    print("Running test suite...")

    start_time = time.time()

    run_test_suite()

    finish_time = (time.time() - start_time)

    print(f"\nELAPSED TIME (seconds): {finish_time:.2f} ({finish_time / 60:.2f} minutes)")

# TODO use proper logging lib for logging
# TODO use rerank across different embeddings sizes (as seen on https://aman.ai/primers/ai/RAG/#figuring-out-the-ideal-chunk-size)
