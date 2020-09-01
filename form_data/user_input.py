# This class handles user data
try:
    from application_logging.logger import App_Logger
    from application_logging import logger
    import pandas as pd
    from pandas.io.json import json_normalize

except ImportError as e:
    print('Import Error occured: ', e)


class user_input:
    def __init__(self, df):
        self.file_object = open("Prediction_Logs/columnValidationLog.txt", 'a+')
        self.log_writer = logger.App_Logger()
        try:
            self.df = pd.DataFrame(json_normalize(df))
            self.log_writer.log(
                self.file_object, 'Input data validated and data frame created for prediction')
        except Exception as e:
            print('Error Occured : ',e)
        
        print(self.df)

    def get_user_input(self, df):
        print('User requesting for input data')
        print(df[['season', 'yr', 'qtrs', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'windspeed',
                  ]])

        print(type(df[['season', 'yr', 'qtrs', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit', 'temp', 'atemp', 'windspeed',
                       ]]))

        print('Before assigning to input variable')
        self.file_object_genlog = open(
            "Prediction_Logs/GeneralLog.txt", 'a+')
        self.log_writer.log(
            self.file_object_genlog, 'Input data validated and data frame created for prediction')
        return df