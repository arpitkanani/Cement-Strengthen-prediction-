import pandas as pd
import sys,os

from src.utils import load_obj
from src.exception import CustomException
from src.logger import logging



class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_obj(file_path=model_path)
            preprocessor=load_obj(file_path=preprocessor_path)
            data_scaled=preprocessor.transform(features)

            prediction=model.predict(data_scaled)

            return prediction
        except Exception as e:
            logging.info("error in predict method")
            raise CustomException(e,sys) #type:ignore
    

class CustomData:
    def __init__(
        self,
        cement: float,
        blast_furance_slag: float,
        fly_ash: float,
        water: float,
        superplasticizer: float,
        coarse_aggregate: float,
        fine_aggregate: float,
        age_in_day: int
    ):
        self.cement = cement
        self.blast_furance_slag = blast_furance_slag
        self.fly_ash = fly_ash
        self.water = water
        self.superplasticizer = superplasticizer
        self.coarse_aggregate = coarse_aggregate
        self.fine_aggregate = fine_aggregate
        self.age_in_day = age_in_day

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={
                'cement':[self.cement],
                'blast_furance_slag':[self.blast_furance_slag],
                'fly_ash':[self.fly_ash],
                'water':[self.water],
                'superplasticizer':[self.superplasticizer],
                'coarse_aggregate':[self.coarse_aggregate],
                'fine_aggregate':[self.fine_aggregate],
                'age_in_day':[self.age_in_day]
            }

            return pd.DataFrame(custom_data_input_dict)
        
        except Exception as e:
            logging.info("error occured in get_data_as_dataframe method in prediction pipeline")
            raise CustomException(e , sys)# type:ignore
            

        