from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
if __name__ == "__main__":
    logging.info("Application has started.")
    
    try:
        data_ingestion=DataIngestion()
        train_data_path,test_data_path,_=data_ingestion.initiate_data_ingestion() 
        #data_transformation_config=DataTransformationConfig()
        data_transformation=DataTransformation()
        X,y,_=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

        model_trainer=ModelTrainer()
        print(model_trainer.initiate_model_trainer(X,y))


    except Exception as e:
        logging.info("custom exception raised")
        raise CustomException(e,sys) # type: ignore

