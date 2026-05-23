import sys
import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

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
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        
        for i in range(len(list(models))):
            model = list(models.values())[i]
                        
            model.fit(X_train, y_train) # Train the model before making predictions

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[list(models.keys())[i]] = test_model_score
                        
            return report
        
    except Exception as e:
        logging.info("Error evaluating model")
        raise CustomException(e, sys)