import yaml
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
import os
import sys
from typing import Any, Dict, List
import numpy as np
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def write_yaml_file(file_path:str,data:dict=None,replace:bool=False):
    """
    Writes a dictionary to a YAML file.
    
    :param file_path: Path to the YAML file.
    :param data: Dictionary to write to the file.
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
                logging.info(f"Existing file at {file_path} removed.")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            yaml.dump(data, file)
        logging.info(f"YAML file written successfully at {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys) 
    

def save_numpy_array_data(file_path: str, array: np.ndarray):
    """
    Saves a NumPy array to a file.
    
    :param file_path: Path to the file where the array will be saved.
    :param array: NumPy array to save.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            np.save(file, array)
        logging.info(f"NumPy array saved successfully at {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def save_object(file_path: str, obj: object):
    """
    Saves an object to a file using pickle.
    
    :param file_path: Path to the file where the object will be saved.
    :param obj: Object to save.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as file:
            pickle.dump(obj, file)
        logging.info(f"Object saved successfully at {file_path}")
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
    
def load_object(file_path: str, ) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        with open(file_path, "rb") as file_obj:
            print(file_obj)
            return pickle.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e
    
def evaluate_models(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray, models: Dict[str, Any], params: Dict[str, List[Any]]) -> Dict[str, float]:
    """
    Evaluates multiple machine learning models and returns their performance metrics.
    
    :param X_train: Training feature set.
    :param y_train: Training labels.
    :param X_test: Testing feature set.
    :param y_test: Testing labels.
    :param models: Dictionary of model names and their instances.
    :param params: Dictionary of model names and their hyperparameters.
    :return: Dictionary with model names as keys and their accuracy scores as values.
    """
    try:
       report: Dict[str, float] = {}
       for i in range(len(list(models))):
           model = list(models.values())[i]
           param = params[list(models.keys())[i]]

           gs = GridSearchCV(model, param, cv=3)
           gs.fit(X_train, y_train)

           model.set_params(**gs.best_params_)
           model.fit(X_train, y_train)

           y_train_pred = model.predict(X_train)
           y_test_pred = model.predict(X_test)

           train_model_score = r2_score(y_true=y_train, y_pred=y_train_pred)
           test_model_score = r2_score(y_true=y_test, y_pred=y_test_pred)

           report[list(models.keys())[i]] = test_model_score
       logging.info(f"Model evaluation report: {report}")
       return report
        
    except Exception as e:
        raise NetworkSecurityException(e, sys) from e