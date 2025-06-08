import os
import sys
from networksecurity.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from networksecurity.entity.artifact_entity import ClassificationMetricArtifact, ModelTrainerArtifact, DataTransformationArtifact
from networksecurity.utils.main_utils.utils import save_object, load_object, load_numpy_array_data,evaluate_models

from networksecurity.utils.ml_utils.metric.classification_metric import get_classification_score
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.utils.ml_utils.model.estimator import NetworkModel

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier , GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from networksecurity.logging.logger import logging
from sklearn.model_selection import GridSearchCV
import mlflow
import dagshub
dagshub.init(repo_owner='sandeep1404', repo_name='NetworkSecurity', mlflow=True)
 
class ModelTrainer:
    def __init__(self, data_transformation_artifact: DataTransformationArtifact, model_trainer_config: ModelTrainerConfig):
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config

    def track_mlflow(self, best_model, classification_metric: ClassificationMetricArtifact):
        with mlflow.start_run():
            f1_score = classification_metric.f1_score
            precision = classification_metric.precision
            recall = classification_metric.recall

            mlflow.log_metric("f1_score", f1_score)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.sklearn.log_model(best_model, "model")


    def train_model(self, X_train, y_train,X_test, y_test):
        models = {
                "Random Forest": RandomForestClassifier(verbose=1),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Boosting": GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost": AdaBoostClassifier(),
            }

        params={
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
                # 'splitter':['best','random'],
                # 'max_features':['sqrt','log2'],
            },
            "Random Forest":{
                # 'criterion':['gini', 'entropy', 'log_loss'],
                
                # 'max_features':['sqrt','log2',None],
                'n_estimators': [8,16,32,128,256]
            },
            "Gradient Boosting":{
                # 'loss':['log_loss', 'exponential'],
                'learning_rate':[.1,.01,.001],
                'subsample':[0.6,0.7,0.75,0.85,0.9],
                # 'criterion':['squared_error', 'friedman_mse'],
                # 'max_features':['auto','sqrt','log2'],
                'n_estimators': [8,16,32,64,128,256]
            },
            "Logistic Regression":{},
            "AdaBoost":{
                'learning_rate':[.1,.01,.001],
                'n_estimators': [8,16,32,128,256]
            }
            
        }

        model_report = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
        best_model_score = max(list(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        best_model = models[best_model_name]
        print(f"Best model found: {best_model_name} with score: {best_model_score}")
        


        y_train_pred = best_model.predict(X_train)
        y_test_pred = best_model.predict(X_test)
        classification_train_metric = get_classification_score(y_true=y_train, y_pred=y_train_pred)
        print(f"Classification metrics training: {classification_train_metric}")

        classification_test_metric = get_classification_score(y_true=y_test, y_pred=y_test_pred)
        print(f"Classification metrics testing: {classification_test_metric}")

        ## Track MLflow
        self.track_mlflow(best_model,classification_train_metric)
        self.track_mlflow(best_model,classification_test_metric)


        preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
        model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
        os.makedirs(model_dir_path, exist_ok=True)

        Network_model = NetworkModel(preprocessor=preprocessor,model=best_model)
        save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=Network_model)
        print(f"Model saved at: {self.model_trainer_config.trained_model_file_path}")

        save_object("final_model/model.pkl", best_model)

        ## model trainer artifact
        model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            trained_metric_artifact=classification_train_metric,
            test_metric_artifact=classification_test_metric
        )
        logging.info(f"Model trainer artifact: {model_trainer_artifact}")

        return model_trainer_artifact



    def initiate_model_trainer(self)->ModelTrainerArtifact:
        try:
            print("Entered the initiate_model_trainer method of ModelTrainer class")
            train_file_path = self.data_transformation_artifact.transformed_train_file_path
            test_file_path = self.data_transformation_artifact.transformed_test_file_path

            train_arr= load_numpy_array_data(file_path=train_file_path)
            test_arr = load_numpy_array_data(file_path=test_file_path)

            X_train, y_train = train_arr[:,:-1], train_arr[:,-1]
            X_test, y_test = test_arr[:,:-1], test_arr[:,-1]
            print("Loaded training and testing data successfully")

            model= self.train_model(X_train, y_train,X_test,y_test)
            print("Model training completed successfully")
            


        except Exception as e:
            raise NetworkSecurityException(e, sys) from e
