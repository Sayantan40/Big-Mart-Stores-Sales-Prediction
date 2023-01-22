
from BigMartSales.exception import SalesException
from BigMartSales.logger import logging
from BigMartSales.entity.config_entity import DataTransformationConfig 
from BigMartSales.entity.artifact_entity import DataIngestionArtifact,\
DataValidationArtifact,DataTransformationArtifact
import sys,os
import numpy as np

from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin


import pandas as pd
from BigMartSales.constants import *
from BigMartSales.utils.util import read_yaml_file,save_numpy_array_data,load_data,save_object



class CustomTransformer(BaseEstimator,TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def fit_transform(self,X,y=None):
        
            try:
                lblEn = LabelEncoder()
                columns = ['item_fat_content','item_type','outlet_size','outlet_location_type','outlet_type']
                data = X.copy()
                for col in columns:
                      data[col]= lblEn.fit_transform(data[col])
                data = data.drop(['item_identifier','outlet_identifier','outlet_establishment_year'],axis=1)
                return np.array(data)

            except Exception as e:
                raise SalesException(e, sys) from e
        
    
    def transform(self,X,y=None):
        return self.fit_transform(X=X)



class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact
                 ):
        
        try:
            
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            
            self.data_transformation_config = data_transformation_config
            
            self.data_ingestion_artifact = data_ingestion_artifact
            
            self.data_validation_artifact = data_validation_artifact

        except Exception as e:
            raise SalesException(e,sys) from e





class DataTransformation:

    def __init__(self, data_transformation_config: DataTransformationConfig,
                 data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_artifact: DataValidationArtifact,
                 ):
        
        try:
            
            logging.info(f"{'>>' * 30}Data Transformation log started.{'<<' * 30} ")
            
            self.data_transformation_config = data_transformation_config
            
            self.data_ingestion_artifact = data_ingestion_artifact
            
            self.data_validation_artifact = data_validation_artifact

            

        except Exception as e:
            raise SalesException(e,sys) from e


    




    def initiate_data_transformation(self)->DataTransformationArtifact:
        try:
            logging.info(f"Obtaining preprocessing object.")
            
            preprocessing_obj = CustomTransformer()


            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path
            

            schema_file_path = self.data_validation_artifact.schema_file_path
            
            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = load_data(file_path=train_file_path, schema_file_path=schema_file_path)
            
            test_df = load_data(file_path=test_file_path, schema_file_path=schema_file_path)

            schema = read_yaml_file(file_path=schema_file_path)

            target_column_name = schema[TARGET_COLUMN_KEY]


            logging.info(f"Splitting input and target feature from training and testing dataframe.")
            input_feature_train_df = train_df.drop(columns=target_column_name,axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=target_column_name,axis=1)
            target_feature_test_df = test_df[target_column_name]
            

            logging.info(f"Applying preprocessing object on training dataframe and testing dataframe")
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            
            

            train_arr = np.c_[input_feature_train_arr,np.array(np.log(target_feature_train_df))]
            test_arr = np.c_[input_feature_test_arr,np.array(np.log(target_feature_test_df))]
            
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir

            train_file_name = os.path.basename(train_file_path).replace(".csv",".npz")
            test_file_name = os.path.basename(test_file_path).replace(".csv",".npz")

            transformed_train_file_path = os.path.join(transformed_train_dir, train_file_name)
            transformed_test_file_path = os.path.join(transformed_test_dir, test_file_name)

            logging.info(f"Saving transformed training and testing array.")
            
            save_numpy_array_data(file_path=transformed_train_file_path,array=train_arr)
            save_numpy_array_data(file_path=transformed_test_file_path,array=test_arr)

            preprocessing_obj_file_path = self.data_transformation_config.preprocessed_object_file_path

            logging.info(f"Saving preprocessing object.")
            save_object(file_path=preprocessing_obj_file_path,obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
                                                                      message="Data transformation successfull.",
                                                                      transformed_train_file_path=transformed_train_file_path,
                                                                      transformed_test_file_path=transformed_test_file_path,
                                                                      preprocessed_object_file_path=preprocessing_obj_file_path)
            logging.info(f"Data transformationa artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise SalesException(e,sys) from e

    def __del__(self):
        logging.info(f"{'>>'*30}Data Transformation log completed.{'<<'*30} \n\n")