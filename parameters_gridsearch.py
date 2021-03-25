# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 10:06:12 2021

@author: Retham
"""

# Number of cross-validations for grid search
cross_val = 10

# Proportion of the data to be used for training in each iteratio
random1Dict = {'a': 0.25, 'b': 0.5}
random2Dict = {'a': 0.65, 'b': 0.95}

# Names of categorical features
categoricalList = ['Categorical1','Categorical2', 'Categorical3']

# Grid search parameters
param_grid_rfc = {'n_estimators': [50, 100, 200, 250],
                  'max_depth': [25, 27, 29, 35, 38],
                  'max_features': ['auto'],
                  'max_leaf_nodes': [None],
                  'min_impurity_decrease': [0, 0.1, 0.3, 0.8],
                  'min_impurity_split': [None],
                  'min_samples_leaf': [1, 3, 7],
                  'min_samples_split': [3, 5, 6, 7],
                  'min_weight_fraction_leaf': [0.0, 0.3, 0.5],
                  'bootstrap': [True],
                  'n_jobs': [1],
                  'oob_score': [False],
                  'random_state': [123],
                  'verbose': [0],
                  'warm_start': [False]}

param_grid_xgbc = {'colsample_bylevel' : [0.5, 0.52],
                   'colsample_bytree' : [0.9, 0.92],
                   'gamma' : [0.0007, 0.0008],
                   'learning_rate' :[0.1, 0.12],
                   'max_delta_step' : [0],
                   'max_depth' : [10, 12],
                   'min_child_weight' : [1],
                   'n_estimators' : [510, 525],
                   'objective' : ['binary:logistic'],
                   'reg_alpha' : [0.09, 0.095],
                   'reg_lambda' : [43, 45],
                   'subsample' : [0.62, 0.65],
                   'booster': ['gbtree']}

param_grid_lgbc = {'num_leaves': [10, 30, 50],
                   'reg_alpha': [0.01, 0.05],
                   'reg_lambda': [0.03, 0.05],
                   'min_data_in_leaf': [1, 2],
                   'learning_rate': [0.01, 0.1],
                   'n_estimators': [100, 500, 2000],
                   'boosting_type': ['gbdt'],
                   'num_threads': [3],
                   'tree_learner': ['serial'],
                   'colsample_bytree': [0.1, 0.5, 0.9],
                   'max_depth': [3, 5, 7],
                   'subsample': [0.1, 0.5],
                   'min_sum_hessian_in_leaf': [1e-3, 0.1]}

param_grid_rfr = {'n_estimators': [50, 100, 200, 250],
                  'max_depth': [25, 27, 29, 35, 38],
                  'max_features': ['auto'],
                  'max_leaf_nodes': [None],
                  'min_impurity_decrease': [0, 0.1, 0.3, 0.8],
                  'min_impurity_split': [None],
                  'min_samples_leaf': [1, 3, 7],
                  'min_samples_split': [3, 5, 6, 7],
                  'min_weight_fraction_leaf': [0.0, 0.3, 0.5],
                  'bootstrap': [True],
                  'n_jobs': [1],
                  'oob_score': [False],
                  'random_state': [123],
                  'verbose': [0],
                  'warm_start': [False]}

param_grid_xgbr = {'colsample_bylevel' : [0.5, 0.52],
                   'colsample_bytree' : [0.9, 0.92],
                   'gamma' : [0.0007, 0.0008],
                   'learning_rate' :[0.1, 0.12],
                   'max_delta_step' : [0],
                   'max_depth' : [10, 12],
                   'min_child_weight' : [1],
                   'n_estimators' : [510, 525],
                   'objective' : ['reg:linear'],
                   'reg_alpha' : [0.09, 0.095],
                   'reg_lambda' : [43, 45],
                   'subsample' : [0.62, 0.65],
                   'booster': ['gbtree']}

param_grid_lgbr = {'num_leaves': [10, 30, 50],
                   'reg_alpha': [0.01, 0.05],
                   'reg_lambda': [0.03, 0.05],
                   'min_data_in_leaf': [1, 2],
                   'learning_rate': [0.01, 0.1],
                   'n_estimators': [100, 500, 2000],
                   'boosting_type': ['gbdt'],
                   'num_threads': [3],
                   'tree_learner': ['serial'],
                   'colsample_bytree': [0.1, 0.5, 0.9],
                   'max_depth': [3, 5, 7],
                   'subsample': [0.1, 0.5],
                   'min_sum_hessian_in_leaf': [1e-3, 0.1]}


