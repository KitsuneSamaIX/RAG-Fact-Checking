"""Test suite.

Run tests and produce reports.
"""

import pandas as pd

from langchain_community.vectorstores import FAISS

from static_search import get_search_results
from common import Fact
from rag_fact_checker import RAGFactChecker
from config import config
from dataset_loader import load_ground_truth_dataset, load_search_engine_results_dataset
from reports import ReportBuilderFor2ClassificationLevels, ReportBuilderFor6ClassificationLevels


def run_test_suite() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    # Load datasets
    ground_truth_df = load_ground_truth_dataset()
    if config.RETRIEVAL_MODE == 'se+vs':
        search_engine_results_df = load_search_engine_results_dataset()

    # Gather IDs
    ids = ground_truth_df.index.to_series()

    # Track execution progress
    n_total = len(ids)
    n_done = 0

    # Initialize ReportBuilder
    match config.CLASSIFICATION_LEVELS:
        case 2:
            rb = ReportBuilderFor2ClassificationLevels()
        case 6:
            rb = ReportBuilderFor6ClassificationLevels()
        case _:
            raise ValueError()

    # If RETRIEVAL_MODE == 'vs' then we can keep using the same RAGFactChecker
    if config.RETRIEVAL_MODE == 'vs':
        vector_store = FAISS.load_local(
            folder_path=config.ALL_EVIDENCE_VECTOR_STORE_PATH,
            embeddings=config.get_embeddings(),
            index_name="FAISS_INDEX_CHUNK_SIZE_1024",  # TODO the '1024' should be a parameter
            allow_dangerous_deserialization=True
        )
        fact_checker = RAGFactChecker.from_vector_store(vector_store)

    for id in ids:
        try:
            print(f"\n\nChecking ID: {id}")

            id_data = _get_fact_and_target(id, ground_truth_df)

            if config.VERBOSE:
                print(f"\nSpeaker: {id_data[0].speaker}")
                print(f"Fact: {id_data[0].text}\n")

            # If RETRIEVAL_MODE == 'se+vs' then we need a new RAGFactChecker each time
            if config.RETRIEVAL_MODE == 'se+vs':
                urls = get_search_results(id, search_engine_results_df)
                if urls.empty:
                    raise RuntimeError(f"No URLs found for ID '{id}'.")
                fact_checker = RAGFactChecker.from_urls(urls)

            pred = fact_checker.check(id_data[0])

            match config.CLASSIFICATION_LEVELS:
                case 2:
                    target = _target_to_bool(id_data[1])
                case 6:
                    target = id_data[1]
                case _:
                    raise ValueError()

            if config.VERBOSE:
                print(f"\nThe target is: {target}\n")

            rb.add_result(target, pred)

        except Exception as e:
            print(f"WARNING: an exception occurred while checking the ID {id}. {e}")
            rb.add_error()

        finally:
            n_done += 1
            print(f"\n\nEXECUTION PROGRESS: done {n_done}/{n_total} ({(n_done / n_total) * 100:.2f}%)")

    # Report
    print("\n\n\n\nREPORT:")
    if rb.are_all_errors():
        print("All ID checks have been aborted due to errors! Check code or network configuration.")
        report = None
        raw_data = None
    else:
        print(f"Checked IDs: {rb.n_total}")
        print(f"Correct answers: {rb.n_correct} ({(rb.n_correct / rb.n_total) * 100:.2f}%)")
        print(f"ID checks aborted due to errors: {rb.n_error}")
        print(f"ID checks completed with undefined predictions: {rb.n_undefined_prediction}")
        report = rb.build()
        raw_data = rb.get_raw_data()

    # Print config
    config.print_config()

    return report, raw_data


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
