# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:13:45 2021

@author: Retham
"""

############################################################################################################################
############################################################ IMPORTS #######################################################
############################################################################################################################

import pandas as pd
import random as rand
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier   
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor 
from xgboost.sklearn import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.externals.joblib import dump

from functions import grid_search_fit, sep_xy_data
from parameters_gridsearch import cross_val, random1Dict, random2Dict, categoricalList
from parameters_gridsearch import param_grid_rfc, param_grid_xgbc, param_grid_lgbc, param_grid_rfr,param_grid_xgbr, param_grid_lgbr

############################################################################################################################
#################################################### DATA EXTRACTION #######################################################
############################################################################################################################

# Load data
data = pd.read_csv('Data/train.csv')
data1 = data[data['y_class'] == 1]
data0 = data[data['y_class'] == 0]

# Sample data randomly (to test sensitivity of forecasts)
rand1 = rand.uniform(**random1Dict)
rand0 = rand.uniform(**random2Dict)
data1samp = data1.sample(frac = rand1)
data0samp = data0.sample(frac = rand0)
data = pd.concat([data1samp,data0samp])

# Separate  X, y and client ID data
X = sep_xy_data(data, range(3,len(data.columns)))
y_class = sep_xy_data(data, range(1,2))
y_reg = sep_xy_data(data, range(2,3))
clientid = pd.DataFrame(data.iloc[:, 0:1].values)
clientid.columns = data.iloc[:, 0:1].columns

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=categoricalList, drop_first = True)

# Training and test data split
X_train, X_test, y_class_train, y_class_test, clientid_train, clientid_test = train_test_split(X, y_class, clientid, test_size = 0.2, random_state = 1234)
X_train, X_test, y_reg_train, y_reg_test, clientid_train, clientid_test = train_test_split(X, y_reg, clientid, test_size = 0.2, random_state = 1234)

# Feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Save scalar weights (Weights folder must exist)
todaysdate = datetime.today().strftime('%Y%m%d')
scaler_full_path = "".join((todaysdate,"_gridsearch_scalar"))
dump(sc, "".join(("Scalar/", scaler_full_path)), compress=True)

############################################################################################################################
############################################################ MODELLING #####################################################
############################################################################################################################ 

############## Random Forest Classification ##############   
    
rfmodel_class = RandomForestClassifier()
best_results_rf_class = grid_search_fit(rfmodel_class, param_grid_rfc, 'f1', max(cross_val, 2), X_train, y_class_train, 'rf_class')

############## XGBoost Classification ####################

xgbmodel_class = XGBClassifier()
best_results_xgb_class = grid_search_fit(xgbmodel_class, param_grid_xgbc, 'f1', max(cross_val, 2), X_train, y_class_train, 'xgb_class')

############## Light GBM Classification ##################

lgbmodel_class = lgb.LGBMClassifier(objective='binary')
best_results_lgb_class = grid_search_fit(lgbmodel_class, param_grid_lgbc, 'f1', max(cross_val, 2), X_train, y_class_train, 'lgb_class')

############## Random Forest Regression ##################

rfmodel_reg = RandomForestRegressor()
best_results_rf_reg = grid_search_fit(rfmodel_reg, param_grid_rfr, None, max(cross_val, 2), X_train, y_reg_train, 'rf_reg')

############## XGBoost Regression ########################

xgbmodel_reg = XGBRegressor()
best_results_xgb_reg = grid_search_fit(xgbmodel_reg, param_grid_xgbr, None, max(cross_val, 2), X_train, y_reg_train, 'xgb_reg')

############## Light GBM Regression ######################

lgbmodel_reg = lgb.LGBMRegressor(objective='binary')
best_results_lgb_reg = grid_search_fit(lgbmodel_reg, param_grid_lgbr, None, max(cross_val, 2), X_train, y_reg_train, 'lgb_reg')

best_results_rf_class.to_csv("".join(("GridSearch/", todaysdate,"_rf_classifier_grid_search.csv")), index_label = 'Model parameter')
best_results_xgb_class.to_csv("".join(("GridSearch/", todaysdate,"_xgb_classifier_grid_search.csv")), index_label = 'Model parameter')
best_results_lgb_class.to_csv("".join(("GridSearch/", todaysdate,"_lgb_classifier_grid_search.csv")), index_label = 'Model parameter')
best_results_rf_reg.to_csv("".join(("GridSearch/", todaysdate,"_rf_regressor_grid_search.csv")), index_label = 'Model parameter')
best_results_xgb_reg.to_csv("".join(("GridSearch/", todaysdate,"_xgb_regressor_grid_search.csv")), index_label = 'Model parameter')
best_results_lgb_reg.to_csv("".join(("GridSearch/", todaysdate,"_lgb_regressor_grid_search.csv")), index_label = 'Model parameter')

