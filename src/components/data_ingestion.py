import os
import sys
import numpy as np
import pandas as pd

from pandas import DataFrame

from pathlib import Path
from src.logging.logger import logging
from src.exception.exception import ChurnErrorException

from src.constant import config
from src.constant.config import PATH_FILE_DATASET
from src.entities.config_entity import DataIngestionConfig
from src.entities.config_artifact import DataIngestionArtifact
from src.utils.main_utils.utils import load_data
from sklearn.model_selection import train_test_split

BASE_DIR = Path.cwd().parent


class DataIngestion:

    def __init__(self, 
                 data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config

        except Exception as e:
            raise ChurnErrorException(e, sys)
        
    def export_data_into_feature_store(self,
                               dataframe: DataFrame
                               ) -> DataFrame:
        
        try:

            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dataframe.to_csv(feature_store_file_path, header=True)

        except Exception as e:
            raise ChurnErrorException(e, sys)
        
    
    def split_data_as_train_test_valid(self, 
                                       dataframe: DataFrame
                                       ):

        try: 

            train_set, test_set = train_test_split(dataframe, 
                                  test_size=self.data_ingestion_config.train_test_split_ratio, 
                                                   random_state=0, shuffle=False)
            
            logging.info("Performed train test split on the dataframe.")


            logging.info("Existed split_data_as_train_test_valid" \
                        "method of DataIngestion class.")

            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)

            logging.info("Exporting train, valid and test set file path")

            train_set.to_csv(self.data_ingestion_config.training_file_path, header=True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, header=True)

            logging.info("Exported train, valid and test set file path")
            
        except Exception as e:
            raise ChurnErrorException(e, sys)
        
    def initiate_data_ingestion(self):


        try:

            file_path = os.path.join(BASE_DIR, 
                                     PATH_FILE_DATASET, 
                                     "datasets_churn.csv")
            
            data = load_data(file_path)

            self.export_data_into_feature_store(data)

            self.split_data_as_train_test_valid(data)

            dataingestionartifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path)
            
            return dataingestionartifact
        
        except Exception as e:
            raise ChurnErrorException(e, sys)