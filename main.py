from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.config_entity import TrainingPipelineConfig, DataIngestionConfig ,DataValidationConfig , DataTransformationConfig
from networksecurity.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
import os
import sys




if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        dataingestion = DataIngestion(data_ingestion_config=dataingestionconfig)
        logging.info("Starting data ingestion process")
        dataingestionartifact = dataingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed successfully")
        # data_validation_config = DataValidationConfig(trainingpipelineconfig)
        # data_validation = DataValidation(dataingestionartifact,data_validation_config)
        # logging.info("Starting data validation process")
        # data_validation_artifact = data_validation.initiate_data_validation()
        # logging.info("Data validation completed successfully")
        data_validation_config=DataValidationConfig(trainingpipelineconfig)
        data_validation=DataValidation(data_validation_config,dataingestionartifact)
        logging.info("Initiate the data Validation")
        data_validation_artifact=data_validation.initiate_data_validation()
        logging.info("data Validation Completed")
        print(data_validation_artifact)


        # data transformation stage 
        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        data_tranformation = DataTransformation(data_transformation_config, data_validation_artifact)
        data_transformation_artifact = data_tranformation.initiate_data_transformation()
        logging.info("Data transformation completed successfully")
        print(data_transformation_artifact)





    except Exception as e:
        logging.error(f"Error occurred during data ingestion: {e}")
        raise NetworkSecurityException(e, sys)