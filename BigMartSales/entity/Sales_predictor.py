import os
import sys

from BigMartSales.exception import SalesException
from BigMartSales.utils.util import load_object

import pandas as pd


class BigMartSalesData:

    def __init__(self,
                
                item_weight:float,
                 
                item_fat_content:str,
                 
                item_visibility:float,
                 
                item_type:str,
                 
                item_mrp:float,
                 
                outlet_size:str,
                 
                outlet_location_type:str,
                 
                outlet_type: str,

                item_outlet_sales: float = None
                 
                 ):
        
        try:
            
            self.item_weight = item_weight
            
            self.item_fat_content = item_fat_content
            
            self.item_visibility = item_visibility
            
            self.item_type = item_type
            
            self.item_mrp = item_mrp
            
            self.outlet_size = outlet_size
            
            self.outlet_location_type = outlet_location_type
            
            self.outlet_type = outlet_type

            self.item_outlet_sales = item_outlet_sales
        
        
        except Exception as e:
            raise SalesException(e, sys) from e

    def get_BigMartSales_input_data_frame(self):

        try:
            
            BigMartSales_input_dict = self.get_BigMartSales_data_as_dict()
            
            return pd.DataFrame(BigMartSales_input_dict)
        
        except Exception as e:
            raise SalesException(e, sys) from e

    def get_BigMartSales_data_as_dict(self):
        
        try:
            
            input_data = {
                "item_weight": [self.item_weight],
                
                "item_fat_content": [self.item_fat_content],
                
                "item_visibility": [self.item_visibility],
                
                "item_type": [self.item_type],
                
                "item_mrp": [self.item_mrp],
                
                "outlet_size": [self.outlet_size],
                
                "outlet_location_type": [self.outlet_location_type],

                "outlet_type": [self.outlet_type]
                }
            
            return input_data
        
        
        except Exception as e:
            raise SalesException(e, sys)


class BigMartSalesPredictor:

    def __init__(self, model_dir: str):
        
        try:
            
            self.model_dir = model_dir
        
        except Exception as e:
            raise SalesException(e, sys) from e

    
    def get_latest_model_path(self):
        
        try:
            
            folder_name = list(map(int, os.listdir(self.model_dir)))
            
            latest_model_dir = os.path.join(self.model_dir, f"{max(folder_name)}")
            
            file_name = os.listdir(latest_model_dir)[0]
            
            latest_model_path = os.path.join(latest_model_dir, file_name)
           
            return latest_model_path
        
        except Exception as e:
            raise BigMartSalesException(e, sys) from e

    def predict(self, X):
        
        try:
            
            model_path = self.get_latest_model_path()
            
            model = load_object(file_path=model_path)
            
            item_outlet_sales= model.predict(X)
            
            return item_outlet_sales
        
        except Exception as e:
            raise SalesException(e, sys) from e