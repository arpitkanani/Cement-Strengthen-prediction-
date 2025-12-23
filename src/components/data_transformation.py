import os ,sys
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder,StandardScaler
from dataclasses import dataclass
from sklearn.compose  import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')


class DataTransformation:

    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_obj(self):
        try:
            logging.info("data transformation initiated: ")
            df=pd.read_csv(r"E:\Data Analysis\Cement-Strenthen\artifacts\raw.csv")
            X=df.iloc[:,:-1]
            num_cols=X.select_dtypes(exclude='object').columns
            
            cat_cols=[]
            logging.info("pipeline initiated")
            
            num_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='median')),
                    ('scaler',StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy='most_frequent'))
                    ,('one_hot_encoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )  
            preprocessor=ColumnTransformer([
                ("num_pipeline",num_pipeline,num_cols),
                ('cat_pipeline',cat_pipeline,cat_cols)
            ])

            logging.info("pipline completed <-->")

            return preprocessor
        except Exception as e:
            logging.info("Error in Transformer method")
    
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Reading of train and test Dataset completed")
            logging.info(f"train dataset: \n{train_df.head(2).to_string()}" )
            logging.info(f'test dataset: \n{test_df.head(2).to_string()}')

            logging.info("obtaining preprocessor object")

            preprocessor_obj=self.get_data_transformer_obj()
            if preprocessor_obj is None:
                raise RuntimeError("preprocessor obj doesn't return anything.")
            
            target_col =train_df.columns[-1]


            input_features_train_df=train_df.drop(columns=target_col,axis=1) # X_train
            target_feature_train_df=train_df[target_col] #y_train

            input_features_test_df=test_df.drop(columns=target_col,axis=1) #X_test
            target_feature_test_df=test_df[target_col] #y_test

            logging.info("applying preprocessing on train and test data set")

            input_features_train_arr=preprocessor_obj.fit_transform(input_features_train_df)# fit_transform on X_train
            input_features_test_arr=preprocessor_obj.transform(input_features_test_df) # transform on X_test

            train_arr=np.c_[
                input_features_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_features_test_arr,np.array(target_feature_test_df)
            ]

            logging.info(f"saved preprocessing object")
            
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )



        except Exception as e:
            logging.info("error occured in initiation phase of data transformatoin")
            raise CustomException(sys,e) # type: ignore

