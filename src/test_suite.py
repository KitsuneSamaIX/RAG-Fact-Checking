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

    # Initialize statistics
    n_total = len(unique_ids)
    n_correct = 0
    n_error = 0
    n_true_positive = 0
    n_false_positive = 0
    n_false_negative = 0

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

            if res is True and target is True:
                n_true_positive += 1

            if res is True and target is False:
                n_false_positive += 1

            if res is False and target is True:
                n_false_negative += 1

        except Exception as e:
            print(f"WARNING: an exception occurred while checking the ID {id}. {e}")
            n_error += 1

    # Report statistics
    print("\n\n\n\nREPORT:")
    if n_total == n_error:
        print("All ID checks have been aborted due to errors! Check code or network configuration.")
    else:
        print(f"Checked IDs: {n_total}")
        print(f"Correct answers: {n_correct} ({(n_correct / n_total) * 100}%)")
        print(f"ID checks aborted due to errors: {n_error}")
        print("\nMETRICS (excluding aborted checks):")
        accuracy = n_correct / (n_total - n_error)
        precision = n_true_positive / (n_true_positive + n_false_positive)
        recall = n_true_positive / (n_true_positive + n_false_negative)
        f1 = 2 * ((precision * recall) / (precision + recall))
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1: {f1}")


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
