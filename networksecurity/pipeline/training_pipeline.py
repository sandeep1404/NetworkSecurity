import os
import sys
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import (
    DataTransformationArtifact,
      DataValidationArtifact, 
      DataIngestionArtifact, 
      ModelTrainerArtifact )


from networksecurity.entity.config_entity import (
    DataTransformationConfig,
    DataValidationConfig,
    DataIngestionConfig,
    ModelTrainerConfig,
    TrainingPipelineConfig,)

from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.cloud.s3_sync import S3Sync
from networksecurity.constants.training_pipeline import TRAINING_BUCKET_NAME
class TrainingPipeline:
    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()
        self.s3_sync = S3Sync()


    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config
            )
            logging.info(f"Data Ingestion Config: {self.data_ingestion_config}")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            logging.info("Starting data ingestion process.")
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion completed and Artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
            
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            self.data_validation_config = DataValidationConfig(self.training_pipeline_config)
            data_validation = DataValidation(data_validation_config=self.data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
            logging.info("Starting data validation process.")
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data Validation completed and Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
            
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            self.start_data_transformation_config = DataTransformationConfig(self.training_pipeline_config)
            data_tranformation = DataTransformation(
                data_transformation_config=self.start_data_transformation_config,
                data_validation_artifact=data_validation_artifact
            )
            data_validation_artifact = data_tranformation.initiate_data_transformation()
            logging.info(f"Data Transformation completed and Artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    ## my local artificat is going to s3 bucket
    def sync_artifact_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/artifact/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.artifact_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    ## local final model is going to s3 bucket
    def sync_saved_models_dir_to_s3(self):
        try:
            aws_bucket_url = f"s3://{TRAINING_BUCKET_NAME}/final_model/{self.training_pipeline_config.timestamp}"
            self.s3_sync.sync_folder_to_s3(folder=self.training_pipeline_config.model_dir, aws_bucket_url=aws_bucket_url)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def start_model_trainer(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            self.start_model_trainer_config = ModelTrainerConfig(self.training_pipeline_config)
            model_trainer = ModelTrainer(
                data_transformation_artifact=data_transformation_artifact,
                model_trainer_config=self.start_model_trainer_config
            )
            logging.info("Starting model training process.")
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model Training completed and Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
 ## run the pipeline by considering all pipeline components
    def run_pipeline(self):
        try:
            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
            self.sync_artifact_dir_to_s3() # artifact dir is going to s3 bucket
            self.sync_saved_models_dir_to_s3() # saved models dir is going to s3 bucket 
            return model_trainer_artifact
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e