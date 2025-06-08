import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from networksecurity.constants.training_pipeline import TARGET_COLUMN , DATA_TRANSFORMATION_IMPUTER_PARAMS
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging
from networksecurity.entity.artifact_entity import DataTransformationArtifact, DataValidationArtifact
from networksecurity.entity.config_entity import DataTransformationConfig, DataValidationConfig
from networksecurity.utils.main_utils.utils import save_numpy_array_data ,save_object

class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"Data Transformation class initialized with config: {data_transformation_config}")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file and returns a DataFrame.
        """
        try:
            logging.info(f"Reading data from {file_path}")
            return pd.read_csv(file_path)
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
        
    def get_data_transformation_object(self) -> Pipeline:
        """
        Creates a data transformation pipeline with KNNImputer.
        """
        try:
            logging.info("Creating data transformation pipeline with KNNImputer.")
            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)
            processor:Pipeline = Pipeline([("imputer", imputer)])
            logging.info("Data transformation pipeline created successfully.")
            return processor
        except Exception as e:
            raise NetworkSecurityException(e, sys) from e

    def initiate_data_transformation(self)->DataTransformationArtifact:
        logging.info("Initiating data transformation process.")
        try:
            logging.info("Starting data transformation process.")
            train_df = DataTransformation.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = DataTransformation.read_data(self.data_validation_artifact.valid_test_file_path)
            logging.info("Data read successfully from validation artifact.")

            ## remove target column from train and test dataframes
            input_features_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]

            target_feature_train_df= target_feature_train_df.replace(-1, 0)

            input_features_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            target_feature_test_df = target_feature_test_df.replace(-1, 0)


            logging.info("Target column removed from train and test dataframes.")

            ## Impute missing values using KNNImputer
            preprocessor = self.get_data_transformation_object()


            logging.info("Fitting and transforming training data using KNNImputer.")
            preprocessor_object = preprocessor.fit(input_features_train_df)
            tranformed_input_train_feature = preprocessor_object.transform(input_features_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_features_test_df)

            train_arr = np.c_[tranformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]


            # save numpy array data
            save_numpy_array_data_train = save_numpy_array_data(file_path=self.data_transformation_config.transformed_train_file_path, array=train_arr)
            logging.info(f"Transformed training data saved at {self.data_transformation_config.transformed_train_file_path}")
            save_numpy_array_data_test = save_numpy_array_data(file_path=self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info(f"Transformed testing data saved at {self.data_transformation_config.transformed_test_file_path}")
            # save preprocessor object
            save_object(file_path=self.data_transformation_config.transformed_object_file_path, obj=preprocessor_object)
            logging.info(f"Preprocessor object saved at {self.data_transformation_config.transformed_object_file_path}")
            logging.info("Data transformation process completed successfully.")

            save_object("final_model/preprocessor.pkl", preprocessor_object)
            logging.info("Preprocessor object saved to final_model/preprocessor.pkl")

            ## preparing the data transformation artifact

            data_transformation_artifact = DataTransformationArtifact(
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,)
            logging.info(f"Data transformation artifact created: {data_transformation_artifact}")

            return data_transformation_artifact

        except Exception as e:
            raise NetworkSecurityException(e, sys) from e