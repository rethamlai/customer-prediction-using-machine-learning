# -*- coding: utf-8 -*-
"""
Created on Wed Mar 24 15:51:37 2021

@author: Retham
"""

############################################################################################################################
############################################################ Imports #######################################################
############################################################################################################################

import pandas as pd
import numpy as np  
from time import time
from datetime import datetime
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from matplotlib import pyplot
import seaborn as sns

############################################################################################################################
############################################################ Functions #####################################################
############################################################################################################################

todaysdate = datetime.today().strftime('%Y%m%d')
xxx= 1
# Separate X, y data
def sep_xy_data(dataframe, position):
    df = pd.DataFrame(dataframe.iloc[:, position].values)
    df.columns = dataframe.iloc[:, position].columns
    df = df.fillna(0)
    return df

# Save accuracy metrics by model     
def score_model_class(y_test, y_pred):
    acc = [accuracy_score(y_test,y_pred)]
    f1 = [f1_score(y_test,y_pred)]
    prec = [precision_score(y_test,y_pred)]
    rec = [recall_score(y_test,y_pred)]
    roc = [roc_auc_score(y_test,y_pred)]
    df = pd.DataFrame([acc, f1, prec, rec, roc])
    df.index = ['Accuracy',
                'F1', 
                'Precision', 
                'Recall', 
                'ROC Area Under the Curve']
    return df

# Save accuracy metrics by model     
def score_model_reg(y_test, y_pred):
    mae = [mean_absolute_error(y_test,y_pred)]
    mse = [mean_squared_error(y_test,y_pred)]     
    val = abs((y_pred['y_pred'] - y_test['y_reg'])/y_test['y_reg'])
    w10 = [sum(1 for x in val if float(x) <=0.1)]
    w20 = [sum(1 for x in val if float(x) >0.1 and float(x) <=0.2)]
    w30 = [sum(1 for x in val if float(x) >0.2 and float(x) <=0.3)]
    p30 = [sum(1 for x in val if float(x) >0.3)]
    w10p = [sum(1 for x in val if float(x) <=0.1)/len(val)]
    w20p = [sum(1 for x in val if float(x) >0.1 and float(x) <=0.2)/len(val)]
    w30p = [sum(1 for x in val if float(x) >0.2 and float(x) <=0.3)/len(val)]
    p30p = [sum(1 for x in val if float(x) >0.3)/len(val)]
    df = pd.DataFrame([mae, mse, w10, w20, w30, p30, w10p, w20p, w30p, p30p])
    df.index = ['Mean Absolute Error',
                'Mean Squared Error',  
                'NumWithin10%', 
                'NumWithin20%',
                'NumWithin30%',
                'Num30%+',
                'PropWithin10%', 
                'PropWithin20%',
                'PropWithin30%',
                'Prop30%+']
    return df

# Fit model to training set then predict on test set
def fit_train_predict_class(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    scoreddf = score_model_class(y_test, y_pred)
    return scoreddf

# Fit model to training set then predict on test set
def fit_train_predict_reg(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = pd.DataFrame(model.predict(X_test), index=y_test.index, columns=['y_pred'])
    scoreddf = score_model_reg(y_test, y_pred)
    return scoreddf

# Save details on sample size and fraction of data used in current iteration

def iteration_details_class(val):
    numberOfAccMetrics = 5
    valreturn = pd.DataFrame(np.transpose([np.repeat(val, numberOfAccMetrics)]))
    valreturn.index = ['Accuracy','F1','Precision','Recall', 'ROC Area Under the Curve']
    return valreturn    

def iteration_details_reg(val):
    numberOfAccMetrics = 10
    valreturn = pd.DataFrame(np.transpose([np.repeat(val, numberOfAccMetrics)]))
    valreturn.index = ['Mean Absolute Error','Mean Squared Error','NumWithin10%','NumWithin20%','NumWithin30%','Num30%+','PropWithin10%','PropWithin20%','PropWithin30%','Prop30%+']
    return valreturn

# Perform model gridsearch then save best params and score
def grid_search_fit(model, param_grid, scoring, cv, X_train, y_train, modelName):    
    start = time()
    gsearch = GridSearchCV(model, 
                           param_grid=param_grid, 
                           scoring=scoring, 
                           cv=cv,
                           verbose=1,
                           n_jobs=4,
                           pre_dispatch = 8,
                           iid=False)    
    gsearch.fit(X_train,y_train)    
    
    # Save gridsearch object (GridSearch folder must exist)
    todaysdate = datetime.today().strftime('%Y%m%d')
    model_filename = "".join(("GridSearch/", todaysdate, "_", modelName,".pkl"))
    joblib.dump(gsearch, model_filename, compress = 3)

    # Output results to dataframe    
    results = pd.DataFrame([])  
    best_params = pd.DataFrame([gsearch.best_params_]).transpose()   
    best_score = pd.DataFrame(np.transpose([np.repeat(gsearch.best_score_, len(best_params))]))   
    best_score.index = best_params.index   
    results = pd.concat([best_params, best_score], axis=1, join = 'inner')
    results.columns = ['Values', 'Best Score']
    elapsed_mins = (time() - start)/60
    print('elapsed minutes: ', elapsed_mins)
    return results

# Get feature importance (SVM - must be a linear kernal, NB - no codes) - Charts folder must exist
def plotImp(model, X, num):
    
    s = str(type(model)).split(".")[-1][:-2]
    
    if s == 'GaussianNB' or s == 'BayesianRidge' or s == 'SVC' or s == 'SVR':
        
        print('Feature importance for: ', s, 'currently unavailable')
        
    elif s == 'LogisticRegression' or s == 'LinearRegression':
        
        feature_imp = pd.DataFrame({'Value':model.coef_[0],'Feature':X.columns})
        pyplot.figure(figsize=(40, 20))
        sns.set(font_scale = 3)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
        pyplot.title(s + ' Features')
        pyplot.tight_layout()
        pyplot.savefig("".join(('Charts/', todaysdate, "_", s, '_feat_imp.png')))
        
    else:
        
        feature_imp = pd.DataFrame({'Value':model.feature_importances_,'Feature':X.columns})
        pyplot.figure(figsize=(40, 20))
        sns.set(font_scale = 3)
        sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False)[0:num])
        pyplot.title(s + ' Features')
        pyplot.tight_layout()
        pyplot.savefig("".join(('Charts/', todaysdate, "_", s, '_feat_imp.png')))








