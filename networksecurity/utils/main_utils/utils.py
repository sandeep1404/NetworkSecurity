import yaml
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.constants.training_pipeline import SCHEMA_FILE_PATH
import os
import sys
from typing import Any, Dict, List
import numpy as np
import pickle


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