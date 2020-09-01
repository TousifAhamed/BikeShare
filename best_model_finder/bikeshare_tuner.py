from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from xgboost import XGBClassifier
import pickle as pikl
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
from sklearn.ensemble import GradientBoostingRegressor
from form_data.user_input import user_input
from form_data.user_input_csv import user_input_csv
from loadModel import loadModel
from xgboost import XGBRegressor
import plotly.graph_objs as go
import plotly
import json


class Model_Finder:
    """
                This class shall  be used to find the model with best accuracy and AUC score.
                Written By: iNeuron Intelligence
                Version: 1.0
                Revisions: None

                """

    def __init__(self, file_object, logger_object):
        self.file_object = file_object
        self.logger_object = logger_object
        self.rb = RandomForestRegressor()
        self.xgb = XGBRegressor(n_estimators=125, random_state=3)
        self.model = GradientBoostingRegressor(
            n_estimators=60, max_depth=1, random_state=1)  # Random_State is SEED value
        self.sv_classifier = SVC()
        self.xgb = XGBClassifier(objective='binary:logistic', n_jobs=-1)
        self.dict = {}
        print('BikeShare Tuner Instance is created')

    def get_best_model(self, x_train, y_train, x_test, y_test):
        """
        Method Name: get_best_params_for_naive_bayes
        Description: get the parameters for the SVM Algorithm which give the best accuracy.
                     Use Hyper Parameter Tuning.
        Output: The model with the best parameters
        On Failure: Raise Exception

        Written By: iNeuron Intelligence
        Version: 1.0
        Revisions: None

                        """
        self.logger_object.log(
            self.file_object, 'Entered the get_best_params_for_svm method of the Model_Finder class')
        try:
            self.rb.fit(x_train, y_train)
            y_predict = self.rb.predict(x_test)
            # rmse = np.sqrt(mean_squared_error(y_test, y_predict))
            rmse = np.sqrt(mean_squared_error(y_predict, y_test))
            model = 'model/RF/RF.P'
            pikl.dump(self.rb, open(model, 'wb'))
            self.dict['RandomForestRegressor'] = rmse
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_svm method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'RandomForestRegression training  failed. Exited the get_best_params_for_svm method of the Model_Finder class')
            raise Exception()

        try:
            self.xgb.fit(x_train, y_train)

            y_predict = self.xgb.predict(x_test)

            # rmse = np.sqrt(mean_squared_error(y_test, y_predict))
            rmse = np.sqrt(mean_squared_error(y_predict, y_test))
            r2 = r2_score(y_test, y_predict)
            self.dict['XGBRegressor'] = rmse
            model = 'model/XGB/XGB.P'
            pikl.dump(self.rb, open(model, 'wb'))
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_svm method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'XGB training  failed. Exited the get_best_params_for_svm method of the Model_Finder class')
            raise Exception()

        try:
            self.model.fit(x_train, y_train)
            print('GB Regression', self.model.score(x_test, y_test))
            y_predicted = self.model.predict(x_test)
            r2 = r2_score(y_test, y_predicted)
            rmse = mean_squared_error(y_test, y_predicted) ** (1/2)
            self.dict['GradientBoostingRegressor'] = rmse
            model = 'model/GB/GB.P'
            pikl.dump(self.rb, open(model, 'wb'))
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_params_for_svm method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'GradientBoost training  failed. Exited the get_best_params_for_svm method of the Model_Finder class')
            raise Exception()

        self.logger_object.log(self.file_object,
                               'Entered the get_best_model method of the Model_Finder class')
        print('From get best model')
        try:
            """
            self.xgboost = self.get_best_params_for_xgboost(train_x, train_y)
            self.prediction_xgboost = self.xgboost.predict(
                test_x)  # Predictions using the XGBoost Model

            # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
            if len(test_y.unique()) == 1:
                self.xgboost_score = accuracy_score(
                    test_y, self.prediction_xgboost)
                self.logger_object.log(
                    self.file_object, 'Accuracy for XGBoost:' + str(self.xgboost_score))  # Log AUC
            else:
                self.xgboost_score = roc_auc_score(
                    test_y, self.prediction_xgboost)  # AUC for XGBoost
                self.logger_object.log(
                    self.file_object, 'AUC for XGBoost:' + str(self.xgboost_score))  # Log AUC

            # create best model for Random Forest
            self.svm = self.get_best_params_for_svm(train_x, train_y)
            # prediction using the SVM Algorithm
            self.prediction_svm = self.svm.predict(test_x)

            # if there is only one label in y, then roc_auc_score returns error. We will use accuracy in that case
            if len(test_y.unique()) == 1:
                self.svm_score = accuracy_score(test_y, self.prediction_svm)
                self.logger_object.log(
                    self.file_object, 'Accuracy for SVM:' + str(self.sv_score))
            else:
                self.svm_score = roc_auc_score(
                    test_y, self.prediction_svm)  # AUC for Random Forest
                self.logger_object.log(
                    self.file_object, 'AUC for SVM:' + str(self.svm_score))

            #comparing the two models
            if(self.svm_score < self.xgboost_score):
                return 'XGBoost', self.xgboost
            else:
                return 'SVM', self.sv_classifier
            """
            print(min(self.dict.items(), key=operator.itemgetter(1))[
                0], ':', round(min(self.dict.items(), key=operator.itemgetter(1))[1], 2))
            result = min(self.dict.items(), key=operator.itemgetter(1))[
                0], ':', round(min(self.dict.items(), key=operator.itemgetter(1))[1], 2)
            return result
        except Exception as e:
            self.logger_object.log(self.file_object,
                                   'Exception occured in get_best_model method of the Model_Finder class. Exception message:  ' + str(
                                       e))
            self.logger_object.log(self.file_object,
                                   'Model Selection Failed. Exited the get_best_model method of the Model_Finder class')
            raise Exception()
