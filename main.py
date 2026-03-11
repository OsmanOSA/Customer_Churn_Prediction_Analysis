import os 
import sys

from src.exception.exception import ChurnErrorException
from src.logging.logger import logging
from src.entities.config_entity import (DataIngestionConfig, DataValidationConfig, TrainingPipelineConfig)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation

if __name__ == "__main__":

    try:

        training_pipeline_config = TrainingPipelineConfig()

        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info("Initiate the data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")

        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info("Initiate the data validation")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info(f"Data validation completed — status: {data_validation_artifact.validation_status}")
    
    except Exception as e:
        raise ChurnErrorException(e, sys)