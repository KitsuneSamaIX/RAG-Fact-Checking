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
    raw_data = []

    print("Running orchestrator...")
    start_time = time.time()

    # for i in [1, 3]:
    for i in range(1, 11):
        # Update config params
        config.TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS = 10
        config.TRUNCATED_RANKING_RETRIEVER_RESULTS = i

        # Run test suite
        report, raw = run_test()
        reports.append(report)
        raw_data.append(raw)

    finish_time = (time.time() - start_time)
    print(f"\nTOTAL ELAPSED TIME (seconds): {finish_time:.2f} ({finish_time / 60:.2f} minutes)")

    # Save report
    report_df = pd.concat(reports, ignore_index=True)
    print(report_df)
    _save_results(report_df, 'metrics.csv')

    # Save raw data
    raw_data_df = pd.concat(raw_data, ignore_index=True)
    _save_results(raw_data_df, 'raw_data.csv')


def _save_results(df: pd.DataFrame, file_name: str):
    timestamp = time.strftime('%Y%m%d-%H%M%S-UTC', time.gmtime())
    file_folder_path = os.path.join(config.RESULTS_PATH, timestamp)
    os.makedirs(file_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    file_path = os.path.join(file_folder_path, file_name)
    df.to_csv(file_path)


def run_test() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    print("Running test suite...")
    start_time = time.time()
    res = run_test_suite()
    finish_time = (time.time() - start_time)
    print(f"\nELAPSED TIME (seconds): {finish_time:.2f} ({finish_time / 60:.2f} minutes)")
    return res
