# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 09:35:11 2021

@author: Retham
"""
# Number of iterations to run training algorithm (each iteration will take a new subset of data based on random1Dict and random2Dict)
numberOfIterations = 5

# Proportion of the data to be used for training in each iteratio
random1Dict = {'a': 0.25, 'b': 0.5}
random2Dict = {'a': 0.65, 'b': 0.95}

# Names of categorical features
categoricalList = ['Categorical1','Categorical2', 'Categorical3']

# Modelling parameters
rfClassifierDict = {'criterion': 'entropy',
                    'bootstrap': True,
                    'max_depth': 30,
                    'max_features': 'auto',
                    'max_leaf_nodes': None,
                    'min_impurity_decrease': 0,
                    'min_impurity_split': None,
                    'min_samples_leaf': 1,
                    'min_samples_split': 3,
                    'min_weight_fraction_leaf': 0.0,
                    'n_estimators': 250,
                    'n_jobs': 1,
                    'oob_score': False,
                    'random_state': 616,
                    'verbose': 0,
                    'warm_start': False}

xgbClassifierDict = {'colsample_bylevel': 0.52,
                     'colsample_bytree': 0.92,
                     'gamma': 0.0008,
                     'learning_rate': 0.12,
                     'max_delta_step': 0,
                     'max_depth': 12,
                     'min_child_weight': 1,
                     'n_estimators': 510,
                     'objective': 'binary:logistic',
                     'reg_alpha': 10.095,
                     'reg_lambda': 45,
                     'subsample': 0.65,
                     'nthread': 4,
                     'scale_pos_weight': 1,
                     'seed': 27}

lgbClassifierDict = {'num_leaves': 10,
                     'reg_alpha': 0.01,
                     'reg_lambda': 0.03,
                     'min_data_in_leaf': 2,
                     'learning_rate': 0.01,
                     'n_estimators': 100,
                     'boosting_type': 'gbdt',
                     'num_threads': 3,
                     'tree_learner': 'serial',
                     'colsample_bytree': 0.5,
                     'max_depth': 5,
                     'subsample': 0.1,
                     'min_sum_hessian_in_leaf': 0.001}

rfRegressorDict = {'bootstrap': True,
                   'max_depth': 30,
                   'max_features': 'auto',
                   'max_leaf_nodes': None,
                   'min_impurity_decrease': 0,
                   'min_impurity_split': None,
                   'min_samples_leaf': 2,
                   'min_samples_split': 3,
                   'min_weight_fraction_leaf': 0.0,
                   'n_estimators': 100,
                   'n_jobs': 1,
                   'oob_score': False,
                   'random_state': 42,
                   'verbose': 0,
                   'warm_start': False}

xgbRegressorDict = {'learning_rate': 0.1,
                    'n_estimators': 140,
                    'max_depth': 5,
                    'min_child_weight': 1,
                    'gamma': 0,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'objective': 'reg:linear',
                    'nthread': 4,
                    'scale_pos_weight': 1,
                    'seed': 27}

lgbRegressorDict = {'num_leaves': 30,
                    'reg_alpha': 0.01,
                    'reg_lambda': 0.03,
                    'min_data_in_leaf': 5,
                    'learning_rate': 0.1,
                    'n_estimators': 2000,
                    'boosting_type': 'gbdt',
                    'num_threads': 3,
                    'tree_learner': 'serial',
                    'colsample_bytree': 0.1,
                    'max_depth': 7,
                    'subsample': 0.1,
                    'min_sum_hessian_in_leaf': 0.001}
























