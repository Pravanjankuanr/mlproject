import sys
import os

import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
import dill

def save_object(file_path, obj):
    try:
        logging.info("Saving object to file: {}".format(file_path))
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
        logging.info("Object saved successfully")
        
    except Exception as e:
        logging.info("Error saving object to file: {}".format(file_path))
        raise CustomException(e, sys)