"""Test suite

Executes tests and produces statistics.
"""

import pandas as pd

from static_search import get_search_results
from fact import Fact
from rag_chain import fact_check
from config import config


def run_test_suite(df: pd.DataFrame):
    ids = df.index.to_series()
    unique_ids = ids.drop_duplicates()

    for id in unique_ids:
        id_data = _get_fact_and_target(id, df)

        print(f"\n\nChecking ID: {id}")
        if config.VERBOSE:
            print(f"\nSpeaker: {id_data[0].speaker}")
            print(f"Fact: {id_data[0].text}\n")

        rs = get_search_results(id, df)
        fact_check(id_data[0], rs)

        # TODO: add target check to compute accuracy
        print(f"\nTHE TARGET IS: {_target_to_bool(id_data[1])}\n")


def _get_fact_and_target(id: str, df: pd.DataFrame) -> tuple[Fact, int]:
    """Gets the fact and the target value of the passed id.
    """
    single_row = df.loc[df.index == id, ['speaker', 'text', 'target']].iloc[0]

    fact = Fact(
        speaker=single_row['speaker'],
        text=single_row['text']
    )
    target = single_row['target']

    return fact, target


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
