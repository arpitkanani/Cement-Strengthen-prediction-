from src.logger import logging
from src.exception import CustomException
import sys
from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    logging.info("Application has started.")
    
    try:
        data_ingestion=DataIngestion()
        train_data_path,test_data_path,_=data_ingestion.initiate_data_ingestion()
        logging.info(f"Data ingestion completed. Train data path: {train_data_path}, Test data path: {test_data_path}, Raw data path: {_}")
    except Exception as e:
        logging.info("custom exception raised")
        raise CustomException(e,sys) # type: ignore

