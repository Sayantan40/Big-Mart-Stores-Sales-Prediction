from BigMartSales.pipeline.pipeline import Pipeline
from BigMartSales.exception import SalesException
from BigMartSales.logger import logging
def main():
    try:
        logging.info('Starting the pipeline...')
        pipeline = Pipeline()
        pipeline.run_pipeline()
    except Exception as e:
        logging.error(f"{e}")
        print(e)
    

if __name__ == '__main__':
    main()

