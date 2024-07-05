"""Main

This is the entrypoint of my RAG for Fake News App.
"""

import pandas as pd

from config import config
from test_suite import run_test

if __name__ == '__main__':
    pd.set_option('mode.copy_on_write', True)
    pd.set_option('display.max_columns', 10)

    df = pd.read_csv(config.AGGR_DATA_PATH)

    if config.USE_SAMPLE:
        unique_ids = df['id'].drop_duplicates()
        unique_ids = unique_ids.sample(config.SAMPLE_SIZE)
        df = pd.merge(df, unique_ids, how='inner', on='id')

    run_test(df)
