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

    n_correct = 0
    n_error = 0

    for id in unique_ids:
        try:
            print(f"\n\nChecking ID: {id}")

            id_data = _get_fact_and_target(id, df)

            if config.VERBOSE:
                print(f"\nSpeaker: {id_data[0].speaker}")
                print(f"Fact: {id_data[0].text}\n")

            urls = get_search_results(id, df)
            res = fact_check(id_data[0], urls)

            target = _target_to_bool(id_data[1])

            if config.VERBOSE:
                print(f"\nThe target is: {target}\n")

            if res == target:
                n_correct += 1

        except Exception as e:
            print(f"WARNING: an exception occurred while checking the ID {id}. {e}")
            n_error += 1

    # Report statistics
    print(f"\n\n\n\nChecked {len(unique_ids)} IDs, correct answers: {n_correct}, ID checks aborted due to errors: {n_error}")


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
