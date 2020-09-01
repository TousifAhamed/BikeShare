import numpy as np
import pandas as pd
# from Prediction_Raw_Data_Validation.predictionDataValidation import #Prediction_Data_validation
from application_logging import logger
# from data_ingestion import data_loader_prediction
# from data_preprocessing import preprocessing
from file_operations import file_methods
import pickle as pikl

class loadModel:
    def __init__(self):
        self.file_object = open("Prediction_Logs/Prediction_Log.txt", 'a+')
        self.log_writer = logger.App_Logger()
        self.log_writer.log(self.file_object, 'Load Model instance is created')

    def predictionFromModel(self,input):
        try:
            # self.pred_data_val.deletePredictionFile()  # deletes the existing prediction file from last run!
            self.log_writer.log(self.file_object, 'Start of Prediction')
            print('Before Logwriter')
            file_loader = file_methods.File_Operation(self.file_object, self.log_writer)
            print('File Loader',file_loader)
            print('file_loader ready ',file_loader)
            bike_model = file_loader.load_model('bike_share_rf_model')

            predval = bike_model.predict(input)
            
        except Exception as ex:
            print('Got some errors')
            print('Error mesage ', ex)
            self.log_writer.log(self.file_object, 'Error occured while running the prediction!! Error:: %s' % ex)
            raise ex
        return predval
