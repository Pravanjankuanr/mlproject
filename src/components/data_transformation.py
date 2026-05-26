import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact', 'preprocessor.pkl')
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_object(self):
        try:
            logging.info("Data Transformation initiated")
            
            numerical_columns = ['writing_score', 'reading_score']            
            categorical_columns = ['gender', 'race_ethnicity', 
                                   'parental_level_of_education', 'lunch',
                                   'test_preparation_course']
            
            logging.info("Numerical columns and Categorical columns are identified")
            
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])
            
            logging.info("Numerical pipeline created")
            
            cat_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ("one_hot_encoder", OneHotEncoder()),
                ('scaler', StandardScaler(with_mean=False))
            ])
            logging.info("Categorical pipeline created")
            
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_columns),
                ('cat_pipeline', cat_pipeline, categorical_columns)
            ])
            
            logging.info("Preprocessor object created")
            logging.info("numerical columns: {}".format(numerical_columns))
            logging.info("categorical columns: {}".format(categorical_columns))
            
            return preprocessor 
                
        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Reading train and test data")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Obtaining preprocessor object")
            preprocessor_obj = self.get_data_transformer_object()
            
            target_column_name = 'math_score'
            numerical_columns = ['writing_score', 'reading_score']
            
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessor object on training and testing data")
            
            input_feature_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor_obj.transform(input_feature_test_df)
            
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
            logging.info("Saved preprocessor object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
            
        except Exception as e:
            logging.info("Error in initiating data transformation")
            raise CustomException(e, sys)