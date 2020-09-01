global bike_model

try:
    from werkzeug.utils import secure_filename
    from flask import Flask, flash, request, redirect, url_for, render_template
    from flask import send_from_directory
    import os
    import glob
    import operator

    import pickle as pikl
    import pandas as pd
    import numpy as np
    import config
    from flask import Response
    from flask import send_file
    from flask_cors import CORS, cross_origin
    from pandas.io.json import json_normalize
    import flask_monitoringdashboard as dashboard
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor
    from form_data.user_input import user_input
    from form_data.user_input_csv import user_input_csv
    from loadModel import loadModel
    from xgboost import XGBRegressor
    import plotly.graph_objs as go
    import plotly, json

except ImportError as ie:
    print('Error ', ie)
finally:
    os.putenv('LANG', 'en_US.UTF-8')
    os.putenv('LC_ALL', 'en_US.UTF-8')

# Create App
app = Flask(__name__)
dashboard.bind(app)
CORS(app)

UPLOAD_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#
CORS(app)

# Run or refresh model
def model():
    print('Model called')
    # Loading model file
    bike_model = pikl.load(
        open(r'model\bike_share_rf_model.P', 'rb'))
    # Returning model file
    return bike_model


# Predict from the model build
@app.route('/predict', methods=['POST', 'GET'])
@cross_origin()
def predict():
    print('Some one called me')
    try:
        if request.form is not None:
            print('From if')
            input_values = request.form
            print('User Form Input Values are as follows: ',input_values)
            # inputX = pd.DataFrame(json_normalize(input_values))
            # print('Data Type of user input is : ', type(inputX))

            # Object Initialization
            user_ip = user_input(input_values)

            if user_ip.df.empty:
                return render_template('home.html')
            else:

                print(80 * '*')
                print('Post user_ip is created', user_ip.df)
                print(80 * '*')

                # print(user_input)
                print('Values from user are as below')
                inpv = user_ip.get_user_input(user_ip.df)
                print('Printing received values')
                print(inpv)

                bike_model = loadModel()

                # predval = bike_model.predict(input)
                print('Bike Model Instantiated', bike_model)

                predval = bike_model.predictionFromModel(inpv)

                print('Pred val is ', predval)

                inpv['predval'] = int(predval)
                print('Int pred value is ',inpv["predval"])

                inpv.columns = ['Season', 'Year', 'Month', 'Quarter', 'Holiday', 'Weekday', 'Workingday', 'Weathersit', 'Temp', 'Atemp',
                             'Windspeed', 'Predcited Result']
                return render_template('predict.html', tables=[inpv.to_html()], titles=inpv.columns.values)
        else:
            return render_template('home.html')
    except ValueError:
        return Response("Error Occurred! %s" %ValueError)
    except KeyError:
        return Response("Error Occurred! %s" %KeyError)
    except Exception as e:
        return Response("Error Occurred! %s" %e)

def qr(x):
    if x in [1,2,3]:
        return 1
    elif x in [4,5,6,7]:
        return 2
    else:
        return 3

def bulkPredict():
    try:
        print('Calling user_input_csv construciton')
        user_ip_csv = user_input_csv()
        df = user_ip_csv.get_user_input_csv()
        print('Data Frame Content')
        return pd.DataFrame(df)
    except Exception as e:
        return Response("Error Occurred! %s" % e)
# Home page that renders for every web call
@app.route("/")
@cross_origin()
def home():
    print('Starting from here')
    # Highlight default landing page
    return render_template("home.html")

# About page that renders for every web call
@app.route("/About")
@cross_origin()
def about():
    print('Going to About Page')
    # Highlight default landing page
    return render_template("About.html")

# EDA Page renders detailed analysis about BikeShare Project
@app.route("/EDA")
@cross_origin()
def eda():
    print('Going to EDA Page')
    return render_template("eda.html")

# Bulkprediction 
@app.route("/Bulkpred")
@cross_origin()
def bulkpred():
    print('Bulk Pred')
    return render_template("bulkpred.html")

@app.route('/bike_share_eda.html')
def show_map():
    return send_file('templates/bike_share_eda.html')


def allowed_file(filename):
    print('Checking file extenstion', '.' in filename and
          filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS)
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    print('From upload file')
    if request.method == 'POST':
        print('from post')
        # check if the post request has the file part
        if 'file' not in request.files:
            print('file in request files')
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        print('printing file ',file, file.filename)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('You have not selected any file, please select csv file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('File name is ', filename)
            flash('File Upload Successful')
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('file uploaded')
            user_data = bulkPredict()
            print('Printing User Data',user_data)
            df = pd.DataFrame(user_data)
            print('Created df from bulkpred')
            print(df)
            print(df.head())
            df.columns = ['Season', 'Year', 'Month', 'Quarter', 'Holiday', 'Weekday', 'Workingday', 'Weathersit', 'Temp', 'Atemp',
                            'Windspeed', 'Predcited Result']
            return render_template('bulkpred.html', tables=[df.to_html()], titles=df.columns.values)

    return render_template('upload.html')


@app.route('/uploads/<filename>')
@cross_origin()
def uploaded_file(filename):

    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route("/retrain", methods=['GET','POST'])
@cross_origin()
def retrain():
    flash('Please upload csv file')    
    return render_template('retrain.html')

@app.route("/retrainPredict", methods=['GET', 'POST'])
@cross_origin()
def retrainPredict():
    print('From upload file')
    if request.method == 'POST':
        print('from post')
        # check if the post request has the file part
        if 'file' not in request.files:
            print('file in request files')
            flash('Please provide csv file')
            return render_template('retrain.html')
        file = request.files['file']
        print('printing file ',file, file.filename)
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('You have not selected any file, please select csv file')
            return render_template('retrain.html')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            print('File name is ', filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print('file uploaded')
            print('from retrain predict page')
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
                    model_data = df
                    print('Post Drop')
                    print(df.head())
                    print(df.shape)

                    X = df
                    Y = input_data['cnt']
                    print(X.shape, Y.shape)
                    x_train, x_test, y_train, y_test = train_test_split(
                        X, Y, test_size=0.20, random_state=3)

                    # print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

                    rb = RandomForestRegressor()
                    rb.fit(x_train, y_train)

                    y_predict = rb.predict(x_test)
                    # rmse = np.sqrt(mean_squared_error(y_test, y_predict))
                    rmse = np.sqrt(mean_squared_error(y_predict, y_test))
                    r2 = r2_score(y_test, y_predict)
                    print('Random Forest Results')
                    print('rmse=', rmse)
                    print(r2*100)
                    dict['RandomForestRegressor'] = rmse

                    #Save model
                    model = 'model/RF/RF.P'
                    pikl.dump(rb, open(model, 'wb'))

                    xgb = XGBRegressor(n_estimators=125, random_state=3)
                    xgb.fit(x_train, y_train)

                    y_predict = xgb.predict(x_test)

                    # rmse = np.sqrt(mean_squared_error(y_test, y_predict))
                    rmse = np.sqrt(mean_squared_error(y_predict, y_test))
                    r2 = r2_score(y_test, y_predict)
                    print('XGBoost Regression Result')
                    print('rmse=', rmse)
                    print(r2*100)
                    dict["XGBRegressor"] = rmse
                    from sklearn.ensemble import GradientBoostingRegressor
                    SEED = 1

                    model = GradientBoostingRegressor(
                        n_estimators=60, max_depth=1, random_state=SEED)
                    model.fit(x_train, y_train)
                    print('GB Regression', model.score(x_test, y_test))
                    y_predicted = model.predict(x_test)
                    r2 = r2_score(y_test, y_predicted)
                    rmse = mean_squared_error(y_test, y_predicted) ** (1/2)
                    print('R2 Score', r2)
                    print('RMSE ', rmse)
                    dict['GradientBoostingRegressor'] = rmse

                    print('\n')
                    print(80*'*')
                    print(min(dict.items(), key=operator.itemgetter(1))[
                        0], ':', round(min(dict.items(), key=operator.itemgetter(1))[1], 2))
                    result = min(dict.items(), key=operator.itemgetter(1))[
                        0], ':', round(min(dict.items(), key=operator.itemgetter(1))[1], 2)
                    flash(result)

                    model_data['Predicted Result'] = rb.predict(model_data).astype(int)
                    model_data.columns = ['Season', 'Year', 'Month', 'Quarter', 'Holiday', 'Weekday', 'Workingday', 'Weathersit', 'Temp', 'Atemp',
                                'Windspeed', 'Predcited Result']
                    return render_template('retrain_n_predict.html', tables=[model_data.to_html()], titles=model_data.columns.values)
                else:
                    print(0)
            return render_template('retrain_n_predict.html')

@app.route("/contact")
@cross_origin()
def contact():
    return render_template('Contact.html')

if __name__ == "__main__":
    print('From main')
    # port = int(os.environ.get('PORT', 8080))
    global bike_model
    app.run(host='0.0.0.0', port=config.PORT, debug=config.DEBUG_MODE)
    # app.run(debug=True)
    # app.run(host='0.0.0.0',port=8080)
