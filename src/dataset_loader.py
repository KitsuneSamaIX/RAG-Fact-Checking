"""Dataset loader.
"""

import os

import pandas as pd

from config import config


def load_ground_truth_dataset() -> pd.DataFrame:
    _, file_extension = os.path.splitext(config.GROUND_TRUTH_DATASET_PATH)
    match file_extension:
        case '.csv':
            df = pd.read_csv(config.GROUND_TRUTH_DATASET_PATH)
        case '.pqt':
            df = pd.read_parquet(config.GROUND_TRUTH_DATASET_PATH)
        case _:
            raise ValueError()

    # Set 'id' as index for performance
    # Note: 'id' column is dropped by default (it becomes the index)
    df.set_index('statement_id', drop=True, inplace=True)
    # df.drop(columns=['Unnamed: 0'], inplace=True)

    # Rename columns to standard names
    df.rename(columns={'statement_text': 'text', 'statement_date': 'date'}, inplace=True)

    # Add empty 'speaker' and 'date' fields on datasets that don't have them
    if 'speaker' not in df.columns:
        df['speaker'] = None
    if 'date' not in df.columns:
        df['date'] = None

    if config.USE_SAMPLE:
        df = df.sample(config.SAMPLE_SIZE)

    if config.VERBOSE:
        print(f"Ground truth dataframe loaded. Shape: {df.shape}")

    return df


def load_search_engine_results_dataset() -> pd.DataFrame:
    df = pd.read_csv(config.SEARCH_ENGINE_RESULTS_DATASET_PATH)

    # Set 'id' as index for performance
    # Note: 'id' column is dropped by default (it becomes the index)
    df.set_index('fact_check_id', drop=True, inplace=True)
    # df.drop(columns=['Unnamed: 0'], inplace=True)

    # Rename columns to standard names
    df.rename(columns={'result_rank': 'rank', 'page_snippet': 'snippet', 'page_url': 'url', 'result_uuid': 'uuid'}, inplace=True)

    if config.VERBOSE:
        print(f"Search engine results dataframe loaded. Shape: {df.shape}")

    return df


# Old function to load data from the 'bert_aggregator_df.csv' dataset, to continue using this dataset you must break it
#  down in two pieces (ground truth and search engine results) like the new datasets.
# def _load_dataset_0() -> pd.DataFrame:
#     df = pd.read_csv(config.DATASET_0_PATH)
#
#     # Set 'id' as index for performance
#     # Note: 'id' column is dropped by default (it becomes the index)
#     df.set_index('id', drop=True, inplace=True)
#     df = df.drop(columns=['Unnamed: 0'])
#
#     if config.USE_SAMPLE:
#         ids = df.index.to_series()
#         unique_ids = ids.drop_duplicates()
#         unique_ids = unique_ids.sample(config.SAMPLE_SIZE)
#         df = pd.merge(df, unique_ids, how='inner', left_index=True, right_index=True)
#         df = df.drop(columns=['id'])
#
#     return df
