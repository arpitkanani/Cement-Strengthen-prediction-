import sys,os
import pandas as pd


from src.logger import logging
from src.exception import CustomException
from src.utils import evaluate_model,save_obj

from dataclasses import dataclass
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR



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
                "KNN Regressor":KNeighborsRegressor(),
                
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
                   # "min_samples_split": [2, 5, 10],
                    #"min_samples_leaf": [1, 2, 4]
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

                "KNN Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    #"metric": ["euclidean", "manhattan"]
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

            save_obj(
                file_path=self.model_trainer_config.trained_model_path
                ,obj=best_model
            )
            predicted=best_model.predict(X_test)
            r2_score=best_model_score
            return r2_score

        except Exception as e:
            logging.info("error occured in model trainer's initiate model training method")
            raise CustomException(sys,e)# type:ignore   
