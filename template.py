import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

list_of_files=[
    "app.py",
    'Dockerfile',
    f'src/components/__init__.py',
    f'src/components/data_ingestion.py',
    f'src/components/data_transformation.py',
    f'src/components/model_trainer.py',
    f'src/components/model_monitoring.py',
    f'src/utils.py',
    f'src/pipelines/__init__.py',
    f'src/pipelines/training_pipeline.py',
    f'src/pipelines/prediction_pipeline.py',
    'requirements.txt',
]

for file_path in list_of_files:
    filepath=Path(file_path)
    fildir,filename=os.path.split(filepath)


    if fildir!="":
        os.makedirs(fildir,exist_ok=True)
        logging.info(f'Creating directory: {fildir} for file: {filename}')


    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath,'w') as fp:
            pass
            logging.info(f'Creating empty file: {filepath}')
    
    else:
        logging.info(f"{filename} is already exists.")