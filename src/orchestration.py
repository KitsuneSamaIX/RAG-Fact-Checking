"""Orchestration.

Orchestrate the execution of tests, aggregate and save reports.
"""

import os
import time
import pandas as pd

from test_suite import run_test_suite
from config import config


def run_orchestrator():
    reports = []

    print("Running orchestrator...")
    start_time = time.time()

    # for i in [1]:
    for i in range(1, 11):
        # Update config params
        config.TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS = i
        config.TRUNCATED_RANKING_RETRIEVER_RESULTS = i

        # Run test suite
        report = run_test()
        reports.append(report)

    finish_time = (time.time() - start_time)
    print(f"\nTOTAL ELAPSED TIME (seconds): {finish_time:.2f} ({finish_time / 60:.2f} minutes)")

    df = pd.concat(reports, ignore_index=True)
    print(df)
    _save_results(df)


def _save_results(df: pd.DataFrame):
    timestamp = time.strftime('%Y%m%d-%H%M%S-UTC', time.gmtime())
    file_folder_path = os.path.join(config.RESULTS_PATH, timestamp)
    os.makedirs(file_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    file_name = 'df' + '.csv'
    file_path = os.path.join(file_folder_path, file_name)
    df.to_csv(file_path)


def run_test() -> pd.DataFrame | None:
    print("Running test suite...")
    start_time = time.time()
    report = run_test_suite()
    finish_time = (time.time() - start_time)
    print(f"\nELAPSED TIME (seconds): {finish_time:.2f} ({finish_time / 60:.2f} minutes)")
    return report
