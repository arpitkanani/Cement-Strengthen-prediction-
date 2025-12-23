import sys,os
from urllib.parse import urlparse
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import mlflow #type: ignore

from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model,save_obj,evalute_model_score

from dataclasses import dataclass
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge

from xgboost import XGBRegressor #type: ignore
from catboost import CatBoostRegressor # type:ignore

import dagshub # type: ignore


try:
    dagshub.init(
        repo_owner='arpitkanani',
        repo_name='Cement-Strengthen-prediction-',
        mlflow=True
    )
except Exception as e:
    print("DAGsHub init failed, continuing without it:", e)


@dataclass
class ModelTrainerConfig:
    trained_model_path=os.path.join("artifacts","model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
    

    def initiate_model_trainer(self,train_array,test_array):
        try:

            logging.info("spliting the train array and test array into dependent and independent feature")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models={
                "Linear Regression":LinearRegression(),
                "Lasso":Lasso(),
                "Ridge":Ridge(),
                "Decision Tree":DecisionTreeRegressor(),
                "Random Forest Regressor":RandomForestRegressor(),
                "Adaboost Regressor":AdaBoostRegressor(),
                "GradiantBoost Regressor":GradientBoostingRegressor(),
                "XGBRegressor":XGBRegressor(),
                "CatBoostRegressor":CatBoostRegressor()
            }
            param = {

                "Linear Regression": {
                #"fit_intercept": [True, False]
                },

                "Lasso": {
                    "alpha": [0.001, 0.01, 0.1, 1, 10],
                    #"max_iter": [1000, 5000]
                },

                "Ridge": {
                    "alpha": [0.1, 1, 10, 50],
                    "solver": ["auto", "svd", "cholesky"]
                },

                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse"],
                    "max_depth": [None, 5,3, 10, 20],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                },

                "Random Forest Regressor": {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'max_features': ['sqrt', 'log2'],
                    'bootstrap': [True, False]
                },

                "Adaboost Regressor": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1]
                },

                "GradiantBoost Regressor": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.01, 0.1],
                    "max_depth": [3, 5],
                    #"subsample": [0.8, 1.0]
                },

                'XGBRegressor' : {
                    "n_estimators": [200, 400],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "max_depth": [4, 6, 8],
                    #"subsample": [0.7, 0.8, 1.0],
                    #"colsample_bytree": [0.7, 0.8, 1.0],
                    "reg_lambda": [1, 5, 10]    
                },
                'CatBoostRegressor':{
                    "iterations": [300, 500,1000],
                    #"learning_rate": [0.03, 0.05, 0.1],
                    "depth": [4, 6, 8],
                    "l2_leaf_reg": [1, 3, 5, 7]
                }
                
            }

            
            model_report:dict=evaluate_model(X_train,y_train,X_test,y_test,models,param)

            print(model_report)
            print("="*8)
            logging.info(f"model report : {model_report}")


            best_model_score=max(sorted(model_report.values()))

            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model=models[best_model_name]

            print(f"best model is :{best_model_name},R2 score : {best_model_score}")
            print("="*12)

            logging.info(f"best Model Found, Model Name :{best_model_name}, R2_score : {best_model_score}")

            print("this is the best model")
            print(best_model_name)

            models_names=list(param.keys())
            
            actual_model=""

            for model in models_names:
                if best_model_name == model:
                    actual_model=actual_model + model
            
            
            best_params=param[actual_model]

            mlflow.set_registry_uri("https://dagshub.com/arpitkanani/Cement-Strengthen-prediction-.mlflow")
            tracking_url_type_store= urlparse(mlflow.get_tracking_uri()).scheme

            #ml flow pipe line
            with mlflow.start_run():
                predicted_qualities=best_model.predict(X_test)

                (score,rmse,mae)=evalute_model_score(y_test,predicted_qualities)

                mlflow.log_params(best_params)

                mlflow.log_metric("rmse",rmse)
                mlflow.log_metric("r2",score)
                mlflow.log_metric("mae",mae)
                

                if tracking_url_type_store != 'file':
                    mlflow.sklearn.log_model(sk_model=best_model,name="model")

                else:
                    mlflow.sklearn.log_model(best_model,'model')

                
            

            #mlflow.autolog()
            if best_model_score<0.6:
                raise CustomException("No best Model Found") # type: ignore
            logging.info(f"not a best model found in training and test dataset")


            save_obj(
                file_path=self.model_trainer_config.trained_model_path
                ,obj=best_model
            )
            
            

        except Exception as e:
            logging.info("error occured in model trainer's initiate model training method")
            raise CustomException(sys,e)# type:ignore   
