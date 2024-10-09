"""Orchestration.

Orchestrate the execution of tests, aggregate and save reports.
"""

import os
import time
import pandas as pd

from test_suite import run_test_suite
from config import config


def run_orchestrator():
    print("Running orchestrator...")

    # Initialize vars
    reports = []
    raw_data = []
    start_time = time.time()
    output_folder_path = _create_output_folder()

    # ##################################################################################################################
    # Run test suite with varying configuration parameters
    #
    # Note:
    #  - make sure the varying configuration parameters are included in the report
    # for llm_name in ['mistral-nemo:12b-instruct-2407-fp16', 'llama3.1:8b-instruct-fp16']:
    #     # Update config params
    #     config.LLM_NAME = llm_name
    #     for dataset_name in ['cikm2024_debona', 'cikm2024_soprano', 'climate_fever', 'feverous']:
    #         # Update config params
    #         config.DATASET_NAME = dataset_name
    #         if dataset_name == 'climate_fever':  # Missing '.pqt' file for this dataset, so we use '.csv' instead
    #             config.GROUND_TRUTH_DATASET_PATH = os.path.join(config.DATASET_PATH_PREFIX, config.DATASET_NAME, 'ground_truth.csv')
    #         else:
    #             config.GROUND_TRUTH_DATASET_PATH = os.path.join(config.DATASET_PATH_PREFIX, config.DATASET_NAME, 'ground_truth.pqt')
    #         config.ALL_EVIDENCE_VECTOR_STORE_PATH = os.path.join(config.DATASET_PATH_PREFIX, config.DATASET_NAME, 'embeddings/512/')
    #         for levels in [2, 6]:
    #             # Update config params
    #             config.CLASSIFICATION_LEVELS = levels
    #             for fill, invert in [(True, False), (False, True), (False, False)]:
    #                 # Update config params
    #                 config.FILL_EVIDENCE = fill
    #                 config.FILL_EVIDENCE_UPPER_LIMIT = 10
    #                 config.INVERT_EVIDENCE = invert
    #                 for i in range(1, 11):
    #                     # Update config params
    #                     config.TRUNCATED_RANKING_RETRIEVER_RESULTS = i
    #
    #                     # Run test suite
    #                     report, raw = run_test()
    #                     reports.append(report)
    #                     raw_data.append(raw)


    config.USE_SAMPLE = True
    config.SAMPLE_SIZE = 3
    for llm_name in ['mistral-nemo:12b-instruct-2407-fp16', 'llama3.1:8b-instruct-fp16']:
        # Update config params
        config.LLM_NAME = llm_name
        for dataset_name in ['cikm2024_debona', 'cikm2024_soprano', 'climate_fever', 'feverous']:
            # Update config params
            config.DATASET_NAME = dataset_name
            if dataset_name == 'climate_fever':  # Missing '.pqt' file for this dataset, so we use '.csv' instead
                config.GROUND_TRUTH_DATASET_PATH = os.path.join(config.DATASET_PATH_PREFIX, config.DATASET_NAME, 'ground_truth.csv')
            else:
                config.GROUND_TRUTH_DATASET_PATH = os.path.join(config.DATASET_PATH_PREFIX, config.DATASET_NAME, 'ground_truth.pqt')
            config.ALL_EVIDENCE_VECTOR_STORE_PATH = os.path.join(config.DATASET_PATH_PREFIX, config.DATASET_NAME, 'embeddings/512/')
            for levels in [2]:
                # Update config params
                config.CLASSIFICATION_LEVELS = levels
                for fill, invert in [(True, False)]:
                    # Update config params
                    config.FILL_EVIDENCE = fill
                    config.FILL_EVIDENCE_UPPER_LIMIT = 10
                    config.INVERT_EVIDENCE = invert
                    for i in [1]:
                        # Update config params
                        config.TRUNCATED_RANKING_RETRIEVER_RESULTS = i

                        # Run test suite
                        report, raw = run_test()
                        reports.append(report)
                        raw_data.append(raw)

    # ##################################################################################################################

    finish_time = (time.time() - start_time)
    print(f"\nTOTAL ELAPSED TIME (seconds): {finish_time:.2f} ({finish_time / 60:.2f} minutes)")

    # Save report
    report_df = pd.concat(reports, ignore_index=True)
    print(report_df)
    _save_dataframe(report_df, output_folder_path, 'metrics.csv')

    # Save raw data
    raw_data_df = pd.concat(raw_data, ignore_index=True)
    _save_dataframe(raw_data_df, output_folder_path, 'raw_data.csv')

    # Save the last config
    config_text = ("IMPORTANT NOTE: this is a snapshot of the last used configuration during tests,"
                   " be sure to check which of the following parameters have been varied during tests.\n\n") + config.get_printable_config()
    _save_text(config_text, output_folder_path, 'config.txt')


def _save_dataframe(df: pd.DataFrame, output_folder_path: str, file_name: str):
    file_path = os.path.join(output_folder_path, file_name)
    df.to_csv(file_path)


def _save_text(text: str, output_folder_path: str, file_name: str):
    file_path = os.path.join(output_folder_path, file_name)
    with open(file_path, 'w') as file:
        file.write(text)


def _create_output_folder() -> str:
    """Returns the path of the output folder created.
    """
    timestamp = time.strftime('%Y%m%d-%H%M%S-UTC', time.gmtime())
    folder_path = os.path.join(config.RESULTS_PATH, timestamp)
    os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist
    return folder_path


def run_test() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    print("Running test suite...")
    start_time = time.time()
    res = run_test_suite()
    finish_time = (time.time() - start_time)
    print(f"\nELAPSED TIME (seconds): {finish_time:.2f} ({finish_time / 60:.2f} minutes)")
    return res
