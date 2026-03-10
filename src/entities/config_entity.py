import os

from datetime import datetime
from src.constant import config


class TrainingPipelineConfig:

    def __init__(self, 
                 timestamp = datetime.now()):
        
        timestamp = timestamp.strftime("%d_%m_%Y_%H_%M_%S")
        self.pipeline_name = config.PIPELINE_NAME
        self.artifact_name = config.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, 
                                         timestamp)
        self.timestamp: str = timestamp 



class DataIngestionConfig:

    def __init__(self,
                 training_config: TrainingPipelineConfig):

        self.data_ingestion_dir: str = os.path.join(
            training_config.artifact_dir,
            config.DATA_INGESTION_DIR_NAME)

        self.feature_store_file_path: str = os.path.join(
            self.data_ingestion_dir,
            config.DATA_INGESTION_FEATURE_STORE_DIR,
            config.FILE_NAME
        )

        self.training_file_path: str = os.path.join(
            self.data_ingestion_dir,
            config.DATA_INGESTION_INGESTED_DIR,
            config.TRAIN_FILE_NAME
        )

        self.testing_file_path: str = os.path.join(
            self.data_ingestion_dir,
            config.DATA_INGESTION_INGESTED_DIR,
            config.TEST_FILE_NAME
        )

        self.submission_file_path: str = os.path.join(
            self.data_ingestion_dir,
            config.DATA_INGESTION_INGESTED_DIR,
            config.SUBMISSION_FILE_NAME
        )

        self.train_test_split_ratio = config.DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
        self.train_valid_split_ratio = config.DATA_INGESTION_TRAIN_VALID_SPLIT_RATIO
        