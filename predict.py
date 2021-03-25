# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 16:13:11 2021

@author: Retham
"""
############################################################################################################################
############################################################ IMPORTS #######################################################
############################################################################################################################

import pandas as pd
import random as rand
import numpy as np
from os import listdir
from os.path import isfile, join
from time import time
from datetime import datetime
from sklearn.externals import joblib
from sklearn.externals.joblib import load

from functions import sep_xy_data, score_model_class, score_model_reg, plotImp
from parameters_train_and_predict import random1Dict, random2Dict, categoricalList

############################################################################################################################
#################################################### DATA EXTRACTION #######################################################
############################################################################################################################

# Load data
data = pd.read_csv('Data/predict.csv')
data1 = data[data['y_class'] == 1]
data0 = data[data['y_class'] == 0]

############################################################################################################################
############################################## Predicting on new data ######################################################
############################################################################################################################

todaysdate = datetime.today().strftime('%Y%m%d')

# Sample data
rand1 = rand.uniform(**random1Dict)
rand0 = rand.uniform(**random2Dict)
data1samp = data1.sample(frac = 1)
data0samp = data0.sample(frac = 1)
data = pd.concat([data1samp,data0samp])
    
# Separate  X, y and client ID data
X = sep_xy_data(data, range(3,len(data.columns)))
y_class = sep_xy_data(data, range(1,2))
y_reg = sep_xy_data(data, range(2,3))
clientid = pd.DataFrame(data.iloc[:, 0:1].values)
clientid.columns = data.iloc[:, 0:1].columns
    
# One-hot encode categorical variables
X = pd.get_dummies(X, columns=categoricalList, drop_first = True)

# Load and apply scaler weights
scaler_filename = "Scalar/20210325_training_scalar_iteration_0"
sc=load(scaler_filename)
X_test = sc.transform(X)
y_test_class = y_class
y_test_reg = y_reg

clientid_test = clientid
X_train = sc.transform(X)
y_train_class = y_class
y_train_reg = y_reg
clientid_train = clientid

# Load paths to model weights in list
mypath = "ModelWeights"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

# Loop through models and calculate classification/regression score + plot top 30 feature importances if available for the model
score_class = pd.DataFrame()
score_reg = pd.DataFrame()
k = 0
start = time()
for i in onlyfiles:
    filename = "".join(('ModelWeights/', onlyfiles[k]))
    loaded_model = joblib.load(open(filename, 'rb'))
    y_pred = pd.DataFrame(loaded_model.predict(X_test))
    y_pred.columns = ['y_pred']
    if "class" in i:
        score = score_model_class(y_test_class, y_pred)
        score.columns = [onlyfiles[k]]
        score_class = pd.concat([score_class, np.transpose(score)], axis=0)
    else:
        score = score_model_reg(y_test_reg, y_pred)
        score.columns = [onlyfiles[k]]
        score_reg = pd.concat([score_reg, np.transpose(score)], axis=0)
    k += 1
    plotImp(loaded_model, X, 30)

# Save results to CSV
score_class.to_csv("".join(("Results/", todaysdate, "_predicted_data_classification.csv")), index = True, index_label = 'Model')
score_reg.to_csv("".join(("Results/", todaysdate, "_predicted_data_regression.csv")), index = True, index_label = 'Model')

mins_elapsed = round((time() - start)/60, 1)
print('Minutes elapsed: ', mins_elapsed)















