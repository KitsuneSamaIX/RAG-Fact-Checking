"""Main module.

This is the entrypoint of the app.
"""

import time
import pandas as pd
from test_suite import run_test_suite

from config import config


def orchestrator():  # TODO use the 'report' to build the visualizations (use seaborn/matplotlib)
    reports = []

    print("Running orchestrator...")
    start_time = time.time()

    for i in [1, 5]:
        # Update config params
        config.TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS = i
        config.TRUNCATED_RANKING_RETRIEVER_RESULTS = i

        # Run test suite
        report = run()
        reports.append(report)

    finish_time = (time.time() - start_time)
    print(f"\nTOTAL ELAPSED TIME (seconds): {finish_time:.2f} ({finish_time / 60:.2f} minutes)")

    df = pd.concat(reports, ignore_index=True)
    print(df)


def run() -> pd.DataFrame | None:
    print("Running test suite...")
    start_time = time.time()
    report = run_test_suite()
    finish_time = (time.time() - start_time)
    print(f"\nELAPSED TIME (seconds): {finish_time:.2f} ({finish_time / 60:.2f} minutes)")
    return report


if __name__ == '__main__':
    pd.set_option('mode.copy_on_write', True)
    pd.set_option('display.max_columns', 10)
    pd.set_option('display.max_rows', 5)

    # run()
    orchestrator()


# TODO use proper logging lib for logging
# TODO use rerank across different embeddings sizes (as seen on https://aman.ai/primers/ai/RAG/#figuring-out-the-ideal-chunk-size)
# TODO fix embeddings error bug
