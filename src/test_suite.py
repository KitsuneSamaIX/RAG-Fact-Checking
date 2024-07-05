"""Test suite

Executes tests and produces statistics.
"""

import pandas as pd

from static_search import get_search_results


def run_test(df: pd.DataFrame):
    unique_ids = df['id'].drop_duplicates()
    for id in unique_ids:
        rs = get_search_results(id, df)
        # TODO..... invoke RAG chain HERE to get results


def _get_target(id: str, df: pd.DataFrame) -> int:
    """Gets the target values of the passed id.
    """
    return df[df.id == id]['target'].iat[0]


def _target_to_bool(target: int) -> bool:
    """Maps each integer target to the corresponding boolean value.

    We say that:
    - values 0,1,2 are FALSE
    - values 3,4,5 are TRUE

    Integer targets correspond to TRUTH-O-METER labels of "politifact.com".
    The mapping is:
    - 5 -> True
    - 4 -> Mostly True
    - 3 -> Half True
    - 2 -> Mostly False
    - 1 -> False
    - 0 -> Pants on Fire
    """
    if target <= 2:
        return False
    else:
        return True
