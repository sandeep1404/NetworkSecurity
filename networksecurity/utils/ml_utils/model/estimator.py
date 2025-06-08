import os
import sys
from networksecurity.constants.training_pipeline import MODEL_FILE_NAME, SAVED_MODEL_DIR


from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

class NetworkModel:
    def __init__(self, preprocessor, model):
        self.preprocessor = preprocessor
        self.model = model
    
    def predict(self, x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_pred = self.model.predict(x_transform)
            return y_pred
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
        

