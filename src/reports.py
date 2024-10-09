"""Functions to compute reports.
"""

from abc import ABC, abstractmethod

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_squared_error, mean_absolute_error
import numpy as np

from config import config


class ReportBuilder(ABC):

    def __init__(self):
        self._n_total = 0
        self._n_correct = 0
        self._n_error = 0
        self._n_undefined_prediction = 0

        self._targets = []
        self._predictions = []

        self._keep_digits = 2

    @property
    def n_total(self):
        return self._n_total

    @property
    def n_correct(self):
        return self._n_correct

    @property
    def n_error(self):
        return self._n_error

    @property
    def n_undefined_prediction(self):
        return self._n_undefined_prediction

    @abstractmethod
    def build(self) -> pd.DataFrame:
        """Builds the report.

        A report consists in a single row of a DataFrame with multiple columns, it summarizes the performance of the system,
         with a certain configuration, compared to the ground truth.

        Returns a DataFrame with a single row and multiple columns.
        """
        pass

    def add_error(self):
        """Increases error count by 1.

        Call this when a test fails with some exception.
        """
        self._n_total += 1
        self._n_error += 1

    def add_result(self, target, prediction):
        """Adds a single result to the report.
        """
        self._n_total += 1

        if target is None:
            raise RuntimeError("The target cannot be None.")

        if prediction is None:
            self._n_undefined_prediction += 1
            return

        if target == prediction:
            self._n_correct += 1

        # Save result
        self._targets.append(target)
        self._predictions.append(prediction)
        assert len(self._targets) == len(self._predictions)

    def are_all_errors(self) -> bool:
        return self._n_total == self._n_error

    def _add_info_data(self, report: dict):
        report['n_total'] = self.n_total
        report['n_correct'] = self.n_correct
        report['n_error'] = self.n_error
        report['n_undefined_prediction'] = self.n_undefined_prediction

    @staticmethod
    def _add_config_data(report: dict):
        if config.RETRIEVAL_MODE == 'se+vs':
            report['TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS'] = config.TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS
        else:
            report['TRUNCATED_RANKING_SEARCH_ENGINE_RESULTS'] = None
        report['TRUNCATED_RANKING_RETRIEVER_RESULTS'] = config.TRUNCATED_RANKING_RETRIEVER_RESULTS
        report['CLASSIFICATION_LEVELS'] = config.CLASSIFICATION_LEVELS
        report['RETRIEVAL_MODE'] = config.RETRIEVAL_MODE
        report['USE_RERANKER'] = config.USE_RERANKER
        report['FILL_EVIDENCE'] = config.FILL_EVIDENCE
        report['INVERT_EVIDENCE'] = config.INVERT_EVIDENCE

    def get_raw_data(self) -> pd.DataFrame:
        """Gets the raw data of this report.

        Returns a DataFrame with multiple rows (one for each result) and multiple columns (one for each attribute).
        """
        raw_data = {
            'target': self._targets,
            'prediction': self._predictions
        }
        self._add_config_data(raw_data)
        return pd.DataFrame(raw_data)


class ReportBuilderFor2ClassificationLevels(ReportBuilder):

    def build(self) -> pd.DataFrame:
        print("\nMETRICS (excluding errors and undefined predictions):")
        # Compute metrics
        accuracy = accuracy_score(self._targets, self._predictions)
        precision = precision_score(self._targets, self._predictions, pos_label=True, average='binary', zero_division=np.nan)
        recall = recall_score(self._targets, self._predictions, pos_label=True, average='binary', zero_division=np.nan)
        f1 = f1_score(self._targets, self._predictions, pos_label=True, average='binary', zero_division=np.nan)

        # Round numbers
        accuracy = round(accuracy, self._keep_digits)
        precision = round(precision, self._keep_digits)
        recall = round(recall, self._keep_digits)
        f1 = round(f1, self._keep_digits)

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
        self._add_info_data(report_data)
        self._add_config_data(report_data)
        return pd.DataFrame(report_data)


class ReportBuilderFor6ClassificationLevels(ReportBuilder):

    def build(self) -> pd.DataFrame:
        print("\nMETRICS (excluding errors and undefined predictions):")
        # Compute metrics
        accuracy = accuracy_score(self._targets, self._predictions)
        mse = mean_squared_error(self._targets, self._predictions)
        mae = mean_absolute_error(self._targets, self._predictions)

        # Round numbers
        accuracy = round(accuracy, self._keep_digits)
        mse = round(mse, self._keep_digits)
        mae = round(mae, self._keep_digits)

        # Log metrics
        print(f"Accuracy: {accuracy}")
        print(f"MSE: {mse}")
        print(f"MAE: {mae}")

        # Return report
        report_data = {
            'accuracy': [accuracy],
            'mse': [mse],
            'mae': [mae]
        }
        self._add_info_data(report_data)
        self._add_config_data(report_data)
        return pd.DataFrame(report_data)


# TODO (for 6 class levels) precision, recall and f1 have to be done by class (see: https://www.evidentlyai.com/classification-metrics/multi-class-metrics)
