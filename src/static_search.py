"""Static search module.

Searches on static scraped data.
"""

import pandas as pd

from config import config


def get_search_results(id: str, df: pd.DataFrame) -> pd.Series:
    """Gets the first N search results (URLs) from the static data and returns them as a Series of URL strings sorted
    by rank.

    The sorting is from higher to lower rank (rank 1 is the highest), example (let 'urls' be the resulting list):
    - urls[0] = rank 1
    - urls[1] = rank 2
    - urls[2] = rank 3
    - etc.

    If config.USE_CACHED_URLS is True then the UUIDs of the cached URLs are returned instead.

    Note:
    - N = config.TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS
    """
    df = df[df.index == id]
    df = df.sort_values('rank')
    if config.USE_CACHED_URLS:
        df = df['uuid']
    else:
        df = df['url']
    df = df.iloc[0:config.TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS]
    return df
