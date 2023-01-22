from flask import Flask, request
import sys
import numpy as np
import pip
from BigMartSales.utils.util import read_yaml_file, write_yaml_file
from matplotlib.style import context
from BigMartSales.logger import logging
from BigMartSales.exception import SalesException
import os, sys
import json
from BigMartSales.config.configuration import Configuartion
from BigMartSales.constants import CONFIG_DIR, TRAINING_PIPELINE_ARTIFACT_DIR_KEY, get_current_time_stamp
from BigMartSales.pipeline.pipeline import Pipeline
from BigMartSales.entity.Sales_predictor import BigMartSalesPredictor, BigMartSalesData
from flask import send_file, abort, render_template


ROOT_DIR = os.getcwd()
LOG_FOLDER_NAME = "logs"
PIPELINE_FOLDER_NAME = "BigMartSales"
SAVED_MODELS_DIR_NAME = "saved_models"
MODEL_CONFIG_FILE_PATH = os.path.join(ROOT_DIR, CONFIG_DIR, "model.yaml")
LOG_DIR = os.path.join(ROOT_DIR, LOG_FOLDER_NAME)
PIPELINE_DIR = os.path.join(ROOT_DIR, PIPELINE_FOLDER_NAME)
MODEL_DIR = os.path.join(ROOT_DIR, SAVED_MODELS_DIR_NAME)


from BigMartSales.logger import get_log_dataframe

BigMartSales_DATA_KEY = "BigMartSales_data"

ITEM_OUTLET_SALES_VALUE_KEY = "item_outlet_sales"   

app = Flask(__name__)


@app.route('/artifact', defaults={'req_path':'BigMartSales'})
@app.route('/artifact/<path:req_path>')
def render_artifact_dir(req_path):
    
    os.makedirs("BigMartSales", exist_ok=True)
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
    
    abs_path = os.path.join(req_path)
    
    print(abs_path)
    
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        
        if ".html" in abs_path:
            
            with open(abs_path, "r", encoding="utf-8") as file:
                
                content = ''
                
                for line in file.readlines():
                    
                    content = f"{content}{line}"
                
                return content
        
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file_name): file_name for file_name in os.listdir(abs_path) if
             "artifact" in os.path.join(abs_path, file_name)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('files.html', result=result)


@app.route('/', methods=['GET', 'POST'])
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        return str(e)


@app.route('/view_experiment_hist', methods=['GET', 'POST'])
def view_experiment_history():
    experiment_df = Pipeline.get_experiments_status()
    context = {
        "experiment": experiment_df.to_html(classes='table table-striped col-12')
    }
    return render_template('experiment_history.html', context=context)


@app.route('/train', methods=['GET', 'POST'])
def train():
    message = ""
    pipeline = Pipeline(config=Configuartion(current_time_stamp=get_current_time_stamp()))
    if not Pipeline.experiment.running_status:
        message = "Training started."
        pipeline.start()
    else:
        message = "Training is already in progress."
    context = {
        "experiment": pipeline.get_experiments_status().to_html(classes='table table-striped col-12'),
        "message": message
    }
    return render_template('train.html', context=context)


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    context = {
        BigMartSales_DATA_KEY: None,
        ITEM_OUTLET_SALES_VALUE_KEY: None
    }

    if request.method == 'POST':
        
        item_weight = float(request.form['item_weight'])
        
        item_fat_content = str(request.form['item_fat_content'])
        if item_fat_content == "Regular":
            item_fat_content = 1.0
        elif item_fat_content == "Low Fat":
            item_fat_content = 0.0
        
        item_visibility = float(request.form['item_visibility'])
        
        item_type = str(request.form['item_type'])
        if item_type == "Fruits and Vegetables":
            item_type = 6.0
        elif item_type == "Household":
            item_type = 9.0
        elif item_type == "Meat":
            item_type = 10.0
        elif item_type == "Snack Foods":
            item_type = 13.0
        elif item_type == "Dairy":
            item_type = 4.0
        elif item_type == "Baking Goods":
            item_type = 0.0
        elif item_type == "Soft Drinks":
            item_type = 14.0
        elif item_type == "Hard Drinks":
            item_type = 7.0
        elif item_type == "Health and Hygiene":
            item_type = 8.0
        elif item_type == "Breads":
            item_type = 1.0
        elif item_type == "Canned":
            item_type = 3.0
        elif item_type == "Frozen Foods":
            item_type = 5.0
        elif item_type == "Seafood":
            item_type = 12.0
        elif item_type == "Starchy Foods":
            item_type = 15.0
        elif item_type == "Breakfast":
            item_type = 2.0
        
        item_mrp = float(request.form['item_mrp'])
        
        outlet_size = str(request.form['outlet_size'])
        if outlet_size == "Medium":
            outlet_size = 1.0
        elif outlet_size == "Small":
            outlet_size = 2.0
        elif outlet_size == "High":
            outlet_size = 0.0
            

        outlet_location_type = str(request.form['outlet_location_type'])
        if outlet_location_type == "Tier 1":
            outlet_location_type = 0.0
        elif outlet_location_type == "Tier 2":
            outlet_location_type = 1.0
        elif outlet_location_type == "Tier 3":
            outlet_location_type = 2.0
        
        
        outlet_type = str(request.form['outlet_type'])
        if outlet_type == "Supermarket Type1":
            outlet_type = 1.0
        elif outlet_type == "Supermarket Type2":
            outlet_type = 2.0
        elif outlet_type == "Supermarket Type3":
            outlet_type = 3.0
        elif outlet_type == "Grocery Store":
            outlet_type = 0.0

        BigMartSales_data = BigMartSalesData(  item_weight = item_weight,
                                   
                                   item_fat_content = item_fat_content,
                                   
                                   item_visibility = item_visibility,
                                   
                                   item_type = item_type,
                                   
                                   item_mrp = item_mrp,
                                   
                                   outlet_size = outlet_size,
                                   
                                   outlet_location_type = outlet_location_type,

                                   outlet_type = outlet_type
                                   
                                   )
        
        BigMartSales_df = BigMartSales_data.get_BigMartSales_input_data_frame()
        
        BigMartSales_predictor = BigMartSalesPredictor(model_dir=MODEL_DIR)
        
        item_outlet_sales = np.exp(BigMartSales_predictor.predict(X = BigMartSales_df))
        
        context = {
            BigMartSales_DATA_KEY: BigMartSales_data.get_BigMartSales_data_as_dict(),
           ITEM_OUTLET_SALES_VALUE_KEY: item_outlet_sales
        }
        
        return render_template('predict.html', context=context)
    
    return render_template("predict.html", context=context)


@app.route('/saved_models', defaults={'req_path': 'saved_models'})
@app.route('/saved_models/<path:req_path>')
def saved_models_dir(req_path):
    
    os.makedirs("saved_models", exist_ok=True)
   
    # Joining the base and the requested path
    print(f"req_path: {req_path}")
   
    abs_path = os.path.join(req_path)
    print(abs_path)
   
    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        
        return send_file(abs_path)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    
    return render_template('saved_models_files.html', result=result)


@app.route("/update_model_config", methods=['GET', 'POST'])
def update_model_config():
    try:
        if request.method == 'POST':
            
            model_config = request.form['new_model_config']
            
            model_config = model_config.replace("'", '"')
            
            print(model_config)
            
            model_config = json.loads(model_config)

            write_yaml_file(file_path=MODEL_CONFIG_FILE_PATH, data=model_config)

        model_config = read_yaml_file(file_path=MODEL_CONFIG_FILE_PATH)
        
        return render_template('update_model.html', result={"model_config": model_config})

    except  Exception as e:
        
        logging.exception(e)
        
        return str(e)


@app.route(f'/logs', defaults={'req_path': f'{LOG_FOLDER_NAME}'})
@app.route(f'/{LOG_FOLDER_NAME}/<path:req_path>')
def render_log_dir(req_path):
    
    os.makedirs(LOG_FOLDER_NAME, exist_ok=True)
    # Joining the base and the requested path
    
    logging.info(f"req_path: {req_path}")
    
    abs_path = os.path.join(req_path)
    print(abs_path)
    # Return 404 if path doesn't exist
    
    if not os.path.exists(abs_path):
       
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        
        log_df = get_log_dataframe(abs_path)
        
        context = {"log": log_df.to_html(classes="table-striped", index=False)}
        
        return render_template('log.html', context=context)

    # Show directory contents
    files = {os.path.join(abs_path, file): file for file in os.listdir(abs_path)}

    result = {
        "files": files,
        "parent_folder": os.path.dirname(abs_path),
        "parent_label": abs_path
    }
    return render_template('log_files.html', result=result)


if __name__ == "__main__":
    app.run(debug=True)