import os
import sys 

import numpy as np
import pandas as pd
import yaml
import pickle
import dill

from datetime import datetime, timedelta
from typing import Tuple, List, Literal
from pandas import DataFrame
from pathlib import Path
from dotenv import load_dotenv

from src.logging.logger import logging
from src.exception.exception import ChurnErrorException

from skopt import BayesSearchCV
from sklearn.metrics import make_scorer, f1_score


load_dotenv()

def load_data(file_path: Path) -> DataFrame:
    """
    Load a dataset from a CSV file.

    The file is read using pandas with automatic 
    delimiter detection.

    Parameters
    ----------
    file_path: Path
        Path to the CSV file to load.

    Returns
    -------
    data: DataFrame
        DataFrame containing the loaded data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    pd.errors.ParserError
        If the file cannot be parsed as a CSV.
    """
    try:

        data = pd.read_csv(file_path, sep=None, 
                           engine="python")
        return data 
    
    except Exception as e:
        raise ChurnErrorException(e, sys)
    

def read_yml_file(file_path: Path) -> dict:
    """
    Read and parse a YAML configuration file.

    Parameters
    ----------
    file_path : Path
        Path to the YAML file to read.

    Returns
    -------
    config : dict
        Dictionary containing the parsed YAML content.

    Raises
    ------
    ChurnErrorException
        If an error occurs while reading 
        or parsing the YAML file.
    
    """

    try: 

        with open(file_path, "rb") as file:
            return yaml.safe_load(file)
    
    except Exception as e:
        raise ChurnErrorException(e, sys)
    

def write_yml_file(file_path: str, 
                   content: object, 
                   replace: bool = False) -> None:
    """
    Write content to a YAML file.

    The directory is created if it does not exist. 
    If `replace` is set to True and the file already exists, 
    it will be removed before writing.

    Parameters
    ----------
    file_path : str
        Path to the YAML file to write.
    content : object
        Python object to serialize and write to the YAML file.
    replace : bool, optional
        Whether to replace the file if it already exists. 
        Default is False.

    Raises
    ------
    ChurnErrorException
        If an error occurs while creating
        directories or writing the file.
    
    """

    try:

        if replace: 
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file)

    except Exception as e:
        raise ChurnErrorException(e, sys)
    
def save_numpy_array_data(
        file_path: str,
        array: np.ndarray
        ) -> None:
    """
    Save a NumPy array to a binary file.

    The target directory is created if it does not exist.
    The array is saved using NumPy's native `.npy` format.

    Parameters
    ----------
    file_path : str
        Path to the file where the NumPy array will be saved.
    array : np.ndarray
        NumPy array to save.

    Raises
    ------
    ChurnErrorException
        If an error occurs while creating the directory or saving the file.
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)

    except Exception as e:
        raise ChurnErrorException(e, sys)
    
def save_object(file_path: str,
                obj: object
                ) -> None:
    """
    Serialize and save a Python object to a file.

    The object is saved using the pickle binary format. 
    The target directory is created if it does not exist.

    Parameters
    ----------
    file_path : str
        Path to the file where the object will be saved.
    obj : object
        Python object to serialize and save.

    Raises
    ------
    ChurnErrorException
        If an error occurs while creating the directory 
        or serializing the object.
    """

    try:

        logging.info("Entered the save_object " \
                     "method of MainUtils class")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info("Exited the save_object method of MainUtils class")

    except Exception as e:
        raise ChurnErrorException(e, sys)
    
def load_object(file_path: str) -> object:
    """
    Load and deserialize a Python object from a file.

    The object is loaded using the pickle binary format.

    Parameters
    ----------
    file_path : str
        Path to the file containing the serialized object.

    Returns
    -------
    obj : object
        Deserialized Python object.

    Raises
    ------
    ChurnErrorException
        If the file does not exist or
        if an error occurs during deserialization.
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist")

        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise ChurnErrorException(e, sys)
    

def evaluate_models(X_train: np.ndarray,
                    y_train: np.ndarray,
                    X_valid: np.ndarray,
                    y_valid: np.ndarray,
                    models: dict,
                    param: dict
                    ) -> dict:
    
    """
    Evaluate multiple classification models 
    using Bayesian hyperparameter optimization.

    Each model in `models` is optimized using 
    over the corresponding search space provided in `param`. 
    After finding the best hyperparameters, the model 
    is trained on the training data and 
    evaluated on the validation data using the macro F1 score.

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix.
    y_train : np.ndarray
        Training labels.
    X_valid : np.ndarray
        Validation feature matrix.
    y_valid : np.ndarray
        Validation labels.
    models : dict
        Dictionary of model name to estimator object.
    param : dict
        Dictionary of model name to hyperparameter search 
        space compatible with 
        BayesSearchCV.

    Returns
    -------
    report : dict
        Dictionary where keys are model names and 
        values are the macro F1 score 
        on the validation set.

    Raises
    ------
    ChurnErrorException
        If an error occurs during model optimization, 
        training, or evaluation.
    """

    try:

        report = {}
        scorer = make_scorer(f1_score, average="macro")

        for model_name, model in models.items():
            search_space = param[model_name]

            bayes_search = BayesSearchCV(
                estimator=model,
                search_spaces=search_space,
                n_iter=32,              
                cv=3,
                scoring=scorer,
                n_jobs=-1,
                random_state=42,
                error_score="raise"
            )

            bayes_search.fit(X_train, y_train)

            best_model = bayes_search.best_estimator_

            # Predictions
            y_train_pred = best_model.predict(X_train)
            y_valid_pred = best_model.predict(X_valid)

            train_model_score = f1_score(y_train, y_train_pred,
                                         average="macro")
            valid_model_score = f1_score(y_valid, y_valid_pred, 
                                         average="macro")

            report[model_name] = valid_model_score

        return report

    except Exception as e:
        raise ChurnErrorException(e, sys)
