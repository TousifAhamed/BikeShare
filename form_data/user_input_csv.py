# This class handles user data
try:
    from application_logging.logger import App_Logger
    from application_logging import logger
    import pandas as pd
    from pandas.io.json import json_normalize
    import os
    import glob
    import pickle as pikl
    import glob
    import pandas as pd
    import pickle
    import numpy as np
    import operator
    import dash
    import dash_core_components as dcc
    import dash_html_components as dhc
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import r2_score, mean_squared_error
    from xgboost import XGBRegressor


except ImportError as e:
    print('Import Error occured: ', e)



class user_input_csv:
    def __init__(self):
        self.file_object = open(
            "Prediction_Logs/GeneralLog.txt", 'a+')
        self.log_writer = logger.App_Logger()
        print('Instance Created')

    def get_user_input_csv(self):
        print('User requesting for input csv data')
        for name in glob.glob(r'data/*.csv'):
            print('From for loop')
            print(name)
            if name.__contains__('.csv'):
                def qr(x):
                    if x in [1, 2, 3]:
                        return 1
                    elif x in [4, 5, 6, 7]:
                        return 2
                    else:
                        return 3
                print(1)
                import pandas as pd
                df = pd.read_csv(name)
                print(df.head())
                df['qtrs'] = df['mnth'].apply(qr)
                df = df.drop(["instant", "dteday", "hum", "cnt",
                              "registered", "casual"], axis=1)
                model = pikl.load(open(r'model/bike_share_rf_model.P', 'rb'))
                pred_val = model.predict(df)
                df['Predicted Result'] = pred_val.astype(np.int)
                print('Returning Dataframe')

                print('Before assigning to input variable')
                self.file_object_genlog = open(
                    "Prediction_Logs/GeneralLog.txt", 'a+')
                self.log_writer.log(
                    self.file_object_genlog, 'Input data validated and data frame created for prediction')
                print('returning Datafame')
                return pd.DataFrame(df)
            else:
                self.logger_object.log(self.file_object,
                                       'csv file not found.')
                print(0)
