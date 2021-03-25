# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 11:11:35 2020

@author: rlai2508
"""
############################################################################################################################
################################################### CHECK AND INSTALL IMPORTS ##############################################
############################################################################################################################

import sys
import subprocess
import pkg_resources

required = {'pandas', 'numpy', 'matplotlib', 'seaborn', 'scikit-learn', 'xgboost', 'lightgbm', 'Keras'}
installed = {pkg.key for pkg in pkg_resources.working_set}
missing = required - installed

if missing:
    python = sys.executable
    subprocess.check_call([python, '-m', 'pip', 'install', *missing], stdout=subprocess.DEVNULL)
    
############################################################################################################################
############################################################ IMPORTS #######################################################
############################################################################################################################
    
# Import packages
import pandas as pd
import os
from time import time
from datetime import datetime
import random as rand
from sklearn.externals import joblib
from sklearn.externals.joblib import dump
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier   
from xgboost.sklearn import XGBClassifier
import lightgbm as lgb
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor 
from xgboost.sklearn import XGBRegressor
from sklearn.linear_model import BayesianRidge

from functions import sep_xy_data, fit_train_predict_class, fit_train_predict_reg, iteration_details_class, iteration_details_reg
from parameters_train_and_predict import random1Dict, random2Dict, categoricalList, numberOfIterations
from parameters_train_and_predict import rfClassifierDict, xgbClassifierDict, lgbClassifierDict, rfRegressorDict, xgbRegressorDict, lgbRegressorDict

############################################################################################################################
##################################################### CHECK AND CREATE FOLDERS #############################################
############################################################################################################################

# Check if folders exists - if not then create the folder. Make sure you're in the correct path first
folders = ['Charts', 'GridSearch', 'Results', 'ModelWeights', 'Scalar', 'Data']
for f in folders:
    if not os.path.exists(f):
        print("".join(('Creating folder \'' ,f , '\'')))
        os.makedirs(f)
    else:
        print("".join(('The folder \'', f, '\' already exists')))
   
############################################################################################################################
#################################################### DATA EXTRACTION #######################################################
############################################################################################################################

# Load data
data = pd.read_csv('Data/train.csv')
data1 = data[data['y_class'] == 1]
data0 = data[data['y_class'] == 0]

############################################################################################################################
#################################################### DATA PREPROCESSING ####################################################
############################################################################################################################

# Create blank dataframe to store accuracy metrics/iteration details
resultstosave_class = pd.DataFrame()
resultstosave_reg = pd.DataFrame()

# For file naming and tracking convenience
iteration_master_name = "v1"

start = time()

# Iterate over data with a different sample each time
for i in range(0,numberOfIterations):
    
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

    # For file naming and tracking convenience
    iteration_name = "".join(("iteration_", str(i)))
    
    # Save scalar weights (Weights folder must exist)
    todaysdate = datetime.today().strftime('%Y%m%d')
    scaler_full_path = "".join((todaysdate,"_training_scalar_", iteration_name))
    dump(sc, "".join(("Scalar/", scaler_full_path)), compress=True)
     
    ############################################################################################################################
    #################################################### CLASSIFICATION MODELLING ##############################################
    ############################################################################################################################ 
      
    ############## Logistic Regression ##############
    
    logClassifier = LogisticRegression()
    
    logdf = fit_train_predict_class(logClassifier, X_train, y_class_train, X_test, y_class_test)
    
    ############## SVM ##############
    
    svmClassifier = svm.SVC(kernel='sigmoid')
    
    svmdf = fit_train_predict_class(svmClassifier, X_train, y_class_train, X_test, y_class_test)
    
    ############## Naive Bayes ##############

    nbClassifier = GaussianNB()
    
    nbdf = fit_train_predict_class(nbClassifier, X_train, y_class_train, X_test, y_class_test)
    
    ############## Random Forest ##############     
    
    rfClassifier = RandomForestClassifier(**rfClassifierDict)
    
    randfdf = fit_train_predict_class(rfClassifier, X_train, y_class_train, X_test, y_class_test)
    
    ############## XGBoost ##############
    
    xgbClassifier = XGBClassifier(**xgbClassifierDict)
    
    xgbdf = fit_train_predict_class(xgbClassifier, X_train, y_class_train, X_test, y_class_test)
    
    ############## Light GBM ##############
    
    lgbClassifier = lgb.LGBMClassifier(**lgbClassifierDict)
    
    lgbdf = fit_train_predict_class(lgbClassifier, X_train, y_class_train, X_test, y_class_test)
    
    ############################################################################################################################
    #################################################### REGRESSION MODELLING ##################################################
    ############################################################################################################################ 
    
    ############## Linear regression ##############

    linRegressor = LinearRegression()

    linregdf = fit_train_predict_reg(linRegressor, X_train, y_reg_train, X_test, y_reg_test)

    ############## SVM ##############
    
    svmRegressor = svm.SVR(kernel='linear')
    
    svmregdf = fit_train_predict_reg(svmRegressor, X_train, y_reg_train, X_test, y_reg_test)

    ############## Naive-Bayes ##############
    
    brRegressor = BayesianRidge()
    
    brregdf = fit_train_predict_reg(brRegressor, X_train, y_reg_train, X_test, y_reg_test)

    ############## Random Forest ##############

    rfRegressor = RandomForestRegressor(**rfRegressorDict)
 
    rfregdf = fit_train_predict_reg(rfRegressor, X_train, y_reg_train, X_test, y_reg_test)
    
    ############## XGBoost ##############
    
    xgbRegressor = XGBRegressor(**xgbRegressorDict)
    
    xgbregdf = fit_train_predict_reg(xgbRegressor, X_train, y_reg_train, X_test, y_reg_test)
    
    ############## Light GBM ##############
    
    lgbRegressor = lgb.LGBMRegressor(**lgbRegressorDict)
    
    lgbregdf = fit_train_predict_reg(lgbRegressor, X_train, y_reg_train, X_test, y_reg_test)
    
    ##########################################################################################################################
    #################################################### SAVE RESULTS ########################################################
    ##########################################################################################################################
    
    lendata1_class = iteration_details_class(len(data1samp))
    
    rand1n_class = iteration_details_class(rand1)
    
    lendata0_class = iteration_details_class(len(data0samp))
    
    rand0n_class = iteration_details_class(rand0)
    
    irep_class = iteration_details_class(i)
     
    lendata1_reg = iteration_details_reg(len(data1samp))
    
    rand1n_reg = iteration_details_reg(rand1)
    
    lendata0_reg = iteration_details_reg(len(data0samp))
    
    rand0n_reg = iteration_details_reg(rand0)
    
    irep_reg = iteration_details_reg(i)
    
    # Concat results from different models and iteration details together
    results_class = pd.concat([logdf, svmdf, nbdf, randfdf, xgbdf, lgbdf, lendata1_class, rand1n_class, lendata0_class, rand0n_class, irep_class], axis=1)
    results_reg = pd.concat([linregdf, svmregdf, brregdf, rfregdf, xgbregdf, lgbregdf, lendata1_reg, rand1n_reg, lendata0_reg, rand0n_reg, irep_reg], axis=1, join='inner')
    
    # Insert column names (models and iteration details)
    results_class.columns = ['Logistic Regression','Support-Vector Machines','Naive-Bayes','Random Forest','XGBoost','Light GBM','n = Yes','Frac. of data (Yes)','n = No','Frac. of data (No)','Iteration']
    results_reg.columns = ['Linear Regression','Support-Vector Machines','Bayesian Ridge','Random Forest','XGBoost','Light GBM','n = Yes','Frac. of data (Yes)','n = No','Frac. of data (No)','Iteration']
    
    # Combine current iteration results with previous iterations
    resultstosave_class = pd.concat([resultstosave_class, results_class])
    resultstosave_reg = pd.concat([resultstosave_reg, results_reg])
    
# Save classifier weights (Weights folder must exist)
def save_model_weights(model, modelName):
    model_filename = "".join(("ModelWeights/", modelName, ".sav"))
    joblib.dump(model, open(model_filename, 'wb'), compress=3) 
    
save_model_weights(logClassifier, 'class_log')
save_model_weights(svmClassifier, 'class_svm')
save_model_weights(nbClassifier, 'class_nb')
save_model_weights(rfClassifier, 'class_rf')
save_model_weights(xgbClassifier, 'class_xgb')
save_model_weights(lgbClassifier, 'class_lgb')

save_model_weights(linRegressor, 'reg_lin')
save_model_weights(svmRegressor, 'reg_svm')
save_model_weights(brRegressor, 'reg_br')
save_model_weights(rfRegressor, 'reg_rf')
save_model_weights(xgbRegressor, 'reg_xgb')
save_model_weights(lgbRegressor, 'reg_lgb')

# Save full results to csv (Results folder must exist)        
resultstosave_class.to_csv("".join(("Results/", todaysdate, "_results_class.csv")), index = True, index_label = 'Metric')
resultstosave_reg.to_csv("".join(("Results/", todaysdate, "_results_reg.csv")), index = True, index_label = 'Metric')

# Combine features + all y_pred
log_pred = logClassifier.predict(sc.transform(X))
svm_pred = svmClassifier.predict(sc.transform(X))
nb_pred = nbClassifier.predict(sc.transform(X))
rf_pred = rfClassifier.predict(sc.transform(X))
xgb_pred = xgbClassifier.predict(sc.transform(X))
lgb_pred = lgbClassifier.predict(sc.transform(X))
feat_analysis_class = pd.concat([clientid, 
                           X, 
                           y_class, 
                           pd.DataFrame(log_pred, columns=['log_pred']),
                           pd.DataFrame(svm_pred, columns=['svm_pred']),
                           pd.DataFrame(nb_pred, columns=['nb_pred']),
                           pd.DataFrame(rf_pred, columns=['rf_pred']),
                           pd.DataFrame(xgb_pred, columns=['xgb_pred']),
                           pd.DataFrame(lgb_pred, columns=['lgb_pred'])], axis=1, join = 'inner')

lin_pred = linRegressor.predict(sc.transform(X))
svm_pred = svmRegressor.predict(sc.transform(X))
br_pred = brRegressor.predict(sc.transform(X))
rf_pred = rfRegressor.predict(sc.transform(X))
xgb_pred = xgbRegressor.predict(sc.transform(X))
lgb_pred = lgbRegressor.predict(sc.transform(X))
feat_analysis_reg = pd.concat([clientid, 
                           X, 
                           y_reg, 
                           pd.DataFrame(lin_pred, columns=['lin_pred']),
                           pd.DataFrame(svm_pred, columns=['svm_pred']),
                           pd.DataFrame(br_pred, columns=['br_pred']),
                           pd.DataFrame(rf_pred, columns=['rf_pred']),
                           pd.DataFrame(xgb_pred, columns=['xgb_pred']),
                           pd.DataFrame(lgb_pred, columns=['lgb_pred'])], axis=1, join = 'inner')
    
feat_analysis_class.to_csv("".join(("Results/", todaysdate, "_feat_analysis_class.csv")), index = False)
feat_analysis_reg.to_csv("".join(("Results/", todaysdate, "_feat_analysis_reg.csv")), index = False)

# Estimate time elapsed to run code in minutes
mins_elapsed = round((time() - start)/60, 1)
print('Minutes elapsed: ', mins_elapsed)




