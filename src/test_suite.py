"""Test suite

Executes tests and produces statistics.
"""

import pandas as pd

from langchain_community.vectorstores import FAISS

from static_search import get_search_results
from fact import Fact
from rag_fact_checker import RAGFactChecker
from config import config
from dataset_loader import load_ground_truth_dataset, load_search_engine_results_dataset


def run_test_suite():
    # Load datasets
    ground_truth_df = load_ground_truth_dataset()
    if config.RETRIEVAL_MODE == 'bing+vs':
        search_engine_results_df = load_search_engine_results_dataset()

    # Gather IDs
    ids = ground_truth_df.index.to_series()

    # Track execution progress
    n_done = 0

    # Initialize statistics
    n_total = len(ids)
    n_correct = 0
    n_error = 0
    n_true_positive = 0
    n_false_positive = 0
    n_false_negative = 0

    # If RETRIEVAL_MODE == 'vs' then we can keep using the same RAGFactChecker
    if config.RETRIEVAL_MODE == 'vs':
        vector_store = FAISS.load_local(config.ALL_EVIDENCE_VECTOR_STORE_PATH, config.get_embeddings())
        fact_checker = RAGFactChecker.from_vector_store(vector_store)

    for id in ids:
        try:
            print(f"\n\nChecking ID: {id}")

            id_data = _get_fact_and_target(id, ground_truth_df)

            if config.VERBOSE:
                print(f"\nSpeaker: {id_data[0].speaker}")
                print(f"Fact: {id_data[0].text}\n")

            # If RETRIEVAL_MODE == 'bing+vs' then we need a new RAGFactChecker each time
            if config.RETRIEVAL_MODE == 'bing+vs':
                urls = get_search_results(id, search_engine_results_df)
                fact_checker = RAGFactChecker.from_urls(urls)

            res = fact_checker.check(id_data[0])

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

        finally:
            n_done += 1
            print(f"\n\nEXECUTION PROGRESS: done {n_done}/{n_total} ({(n_done / n_total) * 100:.2f}%)")

    # Report statistics
    _report_for_2_classification_levels(n_total, n_correct, n_error, n_true_positive, n_false_positive, n_false_negative)


def _report_for_2_classification_levels(n_total, n_correct, n_error, n_true_positive, n_false_positive, n_false_negative):
    # Report statistics
    print("\n\n\n\nREPORT:")
    if n_total == n_error:
        print("All ID checks have been aborted due to errors! Check code or network configuration.")
    else:
        print(f"Checked IDs: {n_total}")
        print(f"Correct answers: {n_correct} ({(n_correct / n_total) * 100:.2f}%)")
        print(f"ID checks aborted due to errors: {n_error}")
        print("\nMETRICS (excluding aborted checks):")
        accuracy = n_correct / (n_total - n_error)
        precision = n_true_positive / (n_true_positive + n_false_positive)
        recall = n_true_positive / (n_true_positive + n_false_negative)
        f1 = 2 * ((precision * recall) / (precision + recall))
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1: {f1:.2f}")


def _get_fact_and_target(id: str, df: pd.DataFrame) -> tuple[Fact, int]:
    """Gets the fact and the target value of the passed id.
    """
    single_row = df.loc[df.index == id, ['speaker', 'text', 'date', 'target']].iloc[0]

    fact = Fact(
        speaker=single_row['speaker'],
        text=single_row['text'],
        date=single_row['date']
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
