"""Functions to compute reports.
"""

import pandas as pd
from config import config


def report_for_2_classification_levels(n_total, n_correct, n_error, n_true_positive, n_false_positive, n_false_negative) -> pd.DataFrame:
    """Returns a DataFrame with a single row and multiple columns.
    """
    print("\nMETRICS (excluding aborted checks):")
    # Compute metrics
    try:
        accuracy = n_correct / (n_total - n_error)
    except ZeroDivisionError:
        accuracy = float('NaN')
    try:
        precision = n_true_positive / (n_true_positive + n_false_positive)
    except ZeroDivisionError:
        precision = float('NaN')
    try:
        recall = n_true_positive / (n_true_positive + n_false_negative)
    except ZeroDivisionError:
        recall = float('NaN')
    try:
        f1 = 2 * ((precision * recall) / (precision + recall))
    except ZeroDivisionError:
        f1 = float('NaN')

    # Round numbers
    keep_digits = 2
    accuracy = round(accuracy, keep_digits)
    precision = round(precision, keep_digits)
    recall = round(recall, keep_digits)
    f1 = round(f1, keep_digits)

    # Log metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")

    # Return report
    report_data = {
        'accuracy': [accuracy],
        'precision': [precision],
        'recall': [recall],
        'f1': [f1]
    }
    _add_config_data(report_data)
    return pd.DataFrame(report_data)


def _add_config_data(report: dict):
    report['TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS'] = [config.TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS]
    report['TRUNCATED_RANKING_RETRIEVER_RESULTS'] = [config.TRUNCATED_RANKING_RETRIEVER_RESULTS]
    report['CLASSIFICATION_LEVELS'] = [config.CLASSIFICATION_LEVELS]
    report['RETRIEVAL_MODE'] = [config.RETRIEVAL_MODE]
    report['USE_RERANKER'] = [config.USE_RERANKER]
