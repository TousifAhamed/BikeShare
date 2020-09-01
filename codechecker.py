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

def qr(x):
    if x in [1, 2, 3]:
        return 1
    elif x in [4, 5, 6, 7]:
        return 2
    else:
        return 3

dict = {}
for name in glob.glob(r'data/*.csv'):
    print(name)

    if name.__contains__('.csv'):
        print(1)
        df = pd.read_csv(name)
        print(df.head())
        input_data = df
        df['qtrs'] = df['mnth'].apply(qr)
        df = df.drop(["instant", "dteday", "hum", "cnt",
                    "registered", "casual"], axis=1)
        print('Post Drop')
        print(df.head())
        print(df.shape)
        # model = pickle.load(open(r'model/bike_share_rf_model.P','rb'))
        # pred_val = model.predict(df)
        # print(pred_val)
        # df['Predicted Result'] = pred_val.astype(np.int)
        # print(df.head())
        
        X = df
        Y = input_data['cnt']
        print(X.shape, Y.shape)
        x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.20,random_state=3)

        # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

        rb = RandomForestRegressor()
        rb.fit(x_train, y_train)

        y_predict = rb.predict(x_test)

        y_predict

        # rmse = np.sqrt(mean_squared_error(y_test, y_predict))
        rmse = np.sqrt(mean_squared_error(y_predict, y_test))
        r2 = r2_score(y_test, y_predict)
        print('Random Forest Results')
        print('rmse=', rmse)
        print(r2*100)
        dict['RandomForestRegressor'] = rmse


        xgb = XGBRegressor(n_estimators=125, random_state=3)
        xgb.fit(x_train, y_train)

        y_predict = xgb.predict(x_test)

        # rmse = np.sqrt(mean_squared_error(y_test, y_predict))
        rmse = np.sqrt(mean_squared_error(y_predict, y_test))
        r2 = r2_score(y_test, y_predict)
        print('XGBoost Regression Result')
        print('rmse=', rmse)
        print(r2*100)
        dict["XGBRegressor"]=rmse
        from sklearn.ensemble import GradientBoostingRegressor
        SEED = 1

        model = GradientBoostingRegressor(n_estimators=60, max_depth=1,random_state=SEED)
        model.fit(x_train, y_train)
        print('GB Regression',model.score(x_test, y_test))
        y_predicted = model.predict(x_test)
        r2 = r2_score(y_test, y_predicted)
        rmse = mean_squared_error(y_test, y_predicted) ** (1/2)
        print('R2 Score', r2)
        print('RMSE ',rmse)
        dict['GradientBoostingRegressor']= rmse

        print('\n')
        print(80*'*')
        print(min(dict.items(), key=operator.itemgetter(1))[
              0], ':', round(min(dict.items(), key=operator.itemgetter(1))[1],2))
    else:   
        print(0)
    
# app = dash.Dash()
