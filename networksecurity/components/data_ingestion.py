from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
import numpy as np
from networksecurity.entity.config_entity import DataIngestionConfig
import os
import sys
import pymongo
from sklearn.model_selection import train_test_split
import pandas as pd
from dotenv import load_dotenv
load_dotenv()
from networksecurity.entity.artifact_entity import DataIngestionArtifact

MANGO_DB_URL = os.getenv("MANGO_DB_URL")

class DataIngestion:
    def __init__(self,data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
            
        except Exception as e:
            raise NetworkSecurityException(e,sys)
    
    def export_data_from_mango_db(self):
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name
            self.mango_clinet = pymongo.MongoClient(MANGO_DB_URL)
            collection = self.mango_clinet[database_name][collection_name]

            df=pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df.drop(columns=["_id"], inplace=True)
            
            df.replace({"na":np.nan}, inplace=True)
            
            return df

        except Exception as e:
            raise NetworkSecurityException(e, sys)
        
    
    def export_data_to_feature_store(self,dataframe: pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            os.makedirs(os.path.dirname(feature_store_file_path), exist_ok=True)
            dataframe.to_csv(feature_store_file_path, index=False)
            logging.info(f"Data exported to feature store at {feature_store_file_path}")
            return dataframe
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            train_set, test_set = train_test_split(
                dataframe, test_size=self.data_ingestion_config.train_test_split_ratio, random_state=42
            )
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path, exist_ok=True)
            dir_path_test = os.path.dirname(self.data_ingestion_config.testing_file_path)
            os.makedirs(dir_path_test, exist_ok=True)
            train_set.to_csv(self.data_ingestion_config.training_file_path, index=False)
            test_set.to_csv(self.data_ingestion_config.testing_file_path, index=False)
            logging.info(f"Train and test data exported to {self.data_ingestion_config.training_file_path} and {self.data_ingestion_config.testing_file_path}")
        except Exception as e:
            raise NetworkSecurityException(e, sys)

    def initiate_data_ingestion(self):
        try:
            dataframe = self.export_data_from_mango_db()
            dataframe = self.export_data_to_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            dataingestionatifact = DataIngestionArtifact(
                train_file_path= self.data_ingestion_config.training_file_path,
                test_file_path= self.data_ingestion_config.testing_file_path,
            )
            return dataingestionatifact
        except Exception as e:
            raise NetworkSecurityException(e, sys)




