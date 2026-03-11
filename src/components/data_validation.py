import sys
import os
import shutil
import numpy as np
import pandas as pd

from pathlib import Path
from src.exception.exception import ChurnErrorException
from src.logging.logger import logging
from src.constant import config
from src.entities.config_artifact import (DataIngestionArtifact,
                                          DataValidationArtifact)
from src.entities.config_entity import DataValidationConfig
from src.constant.config import SCHEMA_FILE_PATH
from src.utils.main_utils.utils import read_yml_file, write_yml_file, load_data

from scipy.stats import ks_2samp


class DataValidation:

    def __init__(self,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yml_file(SCHEMA_FILE_PATH)

        except Exception as e:
            raise ChurnErrorException(e, sys)

    def validate_number_of_columns(self,
                                   filename: Path) -> bool:

        try:
            number_of_columns = len(self._schema_config["columns"])
            logging.info(f"Required number of columns : {number_of_columns}")

            data = load_data(filename)
            logging.info(f"Data frame has columns : {len(data.columns)}")

            if len(data.columns) == number_of_columns:
                return True
            return False

        except Exception as e:
            raise ChurnErrorException(e, sys)

    def validate_column_names(self,
                              filename: Path) -> bool:

        try:
            data = load_data(filename)
            schema_columns = [list(col.keys())[0] for col in self._schema_config["columns"]]
            missing = [col for col in schema_columns if col not in data.columns]

            if missing:
                logging.info(f"Missing columns in {filename}: {missing}")
                return False
            return True

        except Exception as e:
            raise ChurnErrorException(e, sys)

    def detect_dataset_drift(self,
                             base_df: pd.DataFrame,
                             current_df: pd.DataFrame,
                             threshold: float = 0.05) -> bool:

        try:
            status = True
            report = {}

            numeric_columns = [
                list(col.keys())[0]
                for col in self._schema_config["numeric_columns"]
            ]

            for column in numeric_columns:
                if column not in base_df.columns or column not in current_df.columns:
                    continue

                d1 = base_df[column].dropna()
                d2 = current_df[column].dropna()

                is_same_dist = ks_2samp(d1, d2)
                drift_detected = bool(is_same_dist.pvalue < threshold)

                if drift_detected:
                    status = False

                report[column] = {
                    "p_value": float(is_same_dist.pvalue),
                    "drift_detected": drift_detected,
                }

            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yml_file(file_path=drift_report_file_path,
                           content=report,
                           replace=True)
            logging.info(f"Drift report written to {drift_report_file_path}")

            return status

        except Exception as e:
            raise ChurnErrorException(e, sys)


    def initiate_data_validation(self) -> DataValidationArtifact:

        try:

            train_file = self.data_ingestion_artifact.trained_file_path
            test_file = self.data_ingestion_artifact.test_file_path
            submission_file = self.data_ingestion_artifact.submission_file_path

            validation_error_msg = ""

            # --- Validate number of columns ---
            for name, path in [("train", train_file),
                                ("test", test_file),
                                ("submission", submission_file)]:

                if not self.validate_number_of_columns(path):
                    validation_error_msg += (
                        f"Columns count mismatch in {name} file. "
                    )

                if not self.validate_column_names(path):
                    validation_error_msg += (
                        f"Column names mismatch in {name} file. "
                    )

            validation_status = len(validation_error_msg) == 0

            # --- Detect drift (train vs test) ---
            if validation_status:
                train_df = load_data(train_file)
                test_df = load_data(test_file)
                drift_status = self.detect_dataset_drift(train_df, test_df)

                if not drift_status:
                    logging.info("Data drift detected between train and test sets.")
            else:
                logging.info(f"Validation failed: {validation_error_msg}")
                # Still write an empty drift report so the path exists
                os.makedirs(
                    os.path.dirname(
                        self.data_validation_config.drift_report_file_path),
                    exist_ok=True,
                )
                write_yml_file(
                    file_path=self.data_validation_config.drift_report_file_path,
                    content={"error": validation_error_msg},
                    replace=True,
                )

            

            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_submission_file_path=self.data_validation_config.valid_submission_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=self.data_validation_config.invalid_train_file_path,
                invalid_submission_file_path=self.data_validation_config.invalid_submission_file_path,
                invalid_test_file_path=self.data_validation_config.invalid_test_file_path,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact

        except Exception as e:
            raise ChurnErrorException(e, sys)
