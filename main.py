import os 
import sys

from src.exception.exception import ChurnErrorException
from src.logging.logger import logging
from src.entities.config_entity import (DataIngestionConfig, TrainingPipelineConfig, )
from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":

    try:
        
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion =DataIngestion(data_ingestion_config)
        logging.info("Initiate the data ingestion ")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info("Data initiation completed")
    
    except Exception as e:
        ChurnErrorException(e, sys)