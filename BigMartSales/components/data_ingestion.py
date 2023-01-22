import sys,os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from zipfile import ZipFile
from BigMartSales.entity.config_entity import DataIngestionConfig
from BigMartSales.exception import SalesException
from BigMartSales.logger import logging
from BigMartSales.entity.artifact_entity import DataIngestionArtifact
from urllib import request
from sklearn.model_selection import train_test_split




class DataIngestion:

    def __init__(self,data_ingestion_config:DataIngestionConfig ):
        
        try:
            
            logging.info(f"{'>>'*20}Data Ingestion log started.{'<<'*20} ")
            
            self.data_ingestion_config = data_ingestion_config

        
        except Exception as e:
            raise SalesException(e,sys)



    def download_Sales_data(self,) -> str:
        
        try:
            
            download_url = self.data_ingestion_config.dataset_download_url
            

            
            zip_download_dir = self.data_ingestion_config.zip_download_dir
            
            os.makedirs(zip_download_dir,exist_ok=True)

            sales_file_name = "Sales.zip"

            zip_file_path = os.path.join(zip_download_dir, sales_file_name)   

            logging.info(f"Downloading file from :[{download_url}] into :[{zip_file_path}]")
            
            request.urlretrieve(download_url, zip_file_path)
            
            
            logging.info(f"File :[{zip_file_path}] has been downloaded successfully.")
            
            return zip_file_path
            

        
        
        except Exception as e:
            raise SalesException(e,sys) from e



    def extract_zip_file(self,zip_file_path:str):
        
        try:
            
            raw_data_dir = self.data_ingestion_config.raw_data_dir
            
            if os.path.exists(raw_data_dir):
                
                os.remove(raw_data_dir)

            
            os.makedirs(raw_data_dir,exist_ok=True)

            logging.info(f"Extracting zip file: [{zip_file_path}] into dir: [{raw_data_dir}]")
            
            with ZipFile(zip_file_path) as sales_zip_file_obj:
                
                sales_zip_file_obj.extractall(path = raw_data_dir)
            
            logging.info(f"Extraction completed")




        except Exception as e:
            raise SalesException(e,sys) from e





    def split_data_as_train_test(self) -> DataIngestionArtifact:
        
        
        try:
            raw_data_dir = self.data_ingestion_config.raw_data_dir

            file_name = os.listdir(raw_data_dir)[0]

            sales_file_path = os.path.join(raw_data_dir,file_name)


            logging.info(f"Reading csv file: [{sales_file_path}]")
            
            sales_data_frame = pd.read_csv(sales_file_path)
            
            input_feature_df = sales_data_frame.drop(columns="item_outlet_sales",axis=1)
            
            target_feature_df = sales_data_frame["item_outlet_sales"]
            
            

            logging.info(f"Splitting data into train and test")
            
            strat_train_set = None
            
            strat_test_set = None

            X_train, X_test, y_train, y_test = train_test_split(input_feature_df,target_feature_df,test_size=0.2,random_state=42)

           
           
            strat_train_set = pd.concat([X_train,y_train],ignore_index=False,axis=1)
                
            strat_test_set  = pd.concat([X_test,y_test],ignore_index=False,axis=1)

            
            
            train_file_path = os.path.join(self.data_ingestion_config.ingested_train_dir,file_name)

            
            test_file_path = os.path.join(self.data_ingestion_config.ingested_test_dir,file_name)
            
            
            
            if strat_train_set is not None:
                
                os.makedirs(self.data_ingestion_config.ingested_train_dir,exist_ok=True)
                
                logging.info(f"Exporting training dataset to file: [{train_file_path}]")
                
                strat_train_set.to_csv(train_file_path,index=False)

            
            
            if strat_test_set is not None:
                
                os.makedirs(self.data_ingestion_config.ingested_test_dir, exist_ok= True)
                
                logging.info(f"Exporting test dataset to file: [{test_file_path}]")
                
                strat_test_set.to_csv(test_file_path,index=False)
            

            
            data_ingestion_artifact = DataIngestionArtifact(
                                                            train_file_path=train_file_path,
                                                            
                                                            test_file_path=test_file_path,
                                                            
                                                            is_ingested=True,
                                                            
                                                            message=f"Data ingestion completed successfully."
                                                            )
            
            logging.info(f"Data Ingestion artifact:[{data_ingestion_artifact}]")
            
            
            return data_ingestion_artifact
        
        
        except Exception as e:
            raise SalesException(e,sys) from e



    
    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        
        
        try:
            zip_file_path =  self.download_Sales_data()
            self.extract_zip_file(zip_file_path = zip_file_path)
            return self.split_data_as_train_test()
        
        
        except Exception as e:
            raise SalesException(e,sys) from e
    


    def __del__(self):
        
        logging.info(f"{'>>'*20}Data Ingestion log completed.{'<<'*20} \n\n")