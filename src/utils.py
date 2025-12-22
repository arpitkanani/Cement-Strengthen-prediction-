import os,sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import pickle
import numpy as np
import pymysql  
from dotenv import load_dotenv
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.model_selection import GridSearchCV
import dill #type: ignore


load_dotenv()

host=os.getenv('host')
user=os.getenv('user')
password=os.getenv('password')
db=os.getenv('db')

def read_sql_data():
    logging.info("reading data from mysql")
    try:
        mydb=pymysql.connect(
            host=host,
            user=user,
            password=password, # type: ignore
            db=db
        ) # type: ignore

        logging.info("connection established ",mydb)
        df=pd.read_sql_query("select*from cement_feature",mydb)
        return df
    except Exception as e:
        logging.info("error ocuured in reading data from databse")
        raise CustomException(sys,e) # type:ignore
    

def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file_obj:
            pickle.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(sys,e)# type:ignore
    

def evalute_model_score(true,predict):
    score=r2_score(true,predict)
    mae=mean_absolute_error(true,predict)
    rmse=np.sqrt(mean_squared_error(true,predict))

    return score,rmse,mae


def evaluate_model(X_train,y_train,X_test,y_test,models,param):
    try:
        report={}
        for i in range(len(models)):
            model=list(models.values())[i]
            para=param[list(models.keys())[i]]
            #train model
            gs=GridSearchCV(model,para,cv=3)
            gs.fit(X_train,y_train)

            model.set_params(**gs.best_params_)

            model.fit(X_train,y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            #train_model_score=r2_score(X_train,y_train)
            test_model_score=r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        
        return report
    
    except Exception as e:
        logging.info("Error in Model Evaluation")
        raise CustomException(e,sys)  # type: ignore
    

def load_obj(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys) #type:ignore