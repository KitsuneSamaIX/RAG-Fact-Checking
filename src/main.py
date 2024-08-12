"""Main

This is the entrypoint of my RAG for Fake News App.
"""

import time

import pandas as pd

from config import config
from test_suite import run_test_suite

if __name__ == '__main__':
    pd.set_option('mode.copy_on_write', True)
    pd.set_option('display.max_columns', 10)

    df = pd.read_csv(config.AGGR_DATA_PATH)

    # Set 'id' as index for performance
    # Note: 'id' column is dropped by default (it becomes the index)
    df.set_index('id', drop=True, inplace=True)
    df = df.drop(columns=['Unnamed: 0'])

    if config.USE_SAMPLE:
        ids = df.index.to_series()
        unique_ids = ids.drop_duplicates()
        unique_ids = unique_ids.sample(config.SAMPLE_SIZE)
        df = pd.merge(df, unique_ids, how='inner', left_index=True, right_index=True)
        df = df.drop(columns=['id'])

    if config.VERBOSE:
        print("DataFrame loaded and indexed.")

    if config.DEBUG:
        print(f"DataFrame's shape: {df.shape}")
        # print(f"DataFrame's description: {df.describe(include='all')}")

    print("Running test suite...")

    start_time = time.time()

    run_test_suite(df)

    finish_time = (time.time() - start_time)

    print(f"\nELAPSED TIME (seconds): {finish_time:.2f} ({finish_time / 60:.2f} minutes)")
