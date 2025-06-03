from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig
import os
import sys



if __name__ == "__main__":
    try:
        dataingestionconfig = DataIngestionConfig(TrainingPipelineConfig())
        dataingestion = DataIngestion(data_ingestion_config=dataingestionconfig)
        logging.info("Starting data ingestion process")
        dataingestionartifact = dataingestion.initiate_data_ingestion()

    except Exception as e:
        logging.error(f"Error occurred during data ingestion: {e}")
        raise NetworkSecurityException(e, sys)