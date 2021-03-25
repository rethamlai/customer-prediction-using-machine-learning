# Customer prediction using machine-learning
An algorithm designed to take in customer level data for predicting future outcomes using different classification and regression methods

## Types of classification models

- Random forest
- XGBoost
- Logistic regression
- Naive-Bayes
- Support-Vector Machines
- Light GBM

## Types of regression models

- Random forest
- XGBoost
- Linear regression
- Bayes-ridge
- Support-Vector Machines
- Light GBM

## Getting started: setting up the data format

- To start, please make sure you have a data file in the format of a CSV in the "Data" folder. Please see the sample data provided.
- The data needs to be in the format of a 'flat' file. This means each row must represent a unique individual/entity and the columns represent that individuals/entities features.
- The data should be split between training and prediction (note: the training data will be further split between training and testing, and the predicting data will be used for validation). It has been designed this way so the file "predict.py" can be used to make predictions without having to train and optimise the model parameters each time.
- The first column of the data files are reserved for the client identifier.
- The second column of the data files are reserved for the classification outcome variable.
- The third column of the data files are reserved for the regression outcome variable.
- If there is no classification outcome variable or regression outcome variable then just randomly fill this column with values. The model will run both classification and regression models, but you can just consider the results for the outcome variable you do have.
- If you have categorical variables, please change the names in "parameters_gridsearch.py" and "parameters_train_and_predict.py" to what you have named your categorical variables in the data.

## Getting started: how to use the files

**gridsearch.py**

- This Python file is used to gridsearch model parameters for the random forest model, XGBoost model and Light GBM model. The algorithm will take a subsample of the training data for grid searching. To change the parameter search space or the subsample of data used for grid searching , please see "parameters_gridsearch.py".

**train.py**

- This Python file is used to train the various classification and regression models using the training data. Users can first run "gridsearch.py" to find the optimal parameters. Once found, users can input the results into "parameters_train_and_predict.py" for training. The algorithm will take a subsample of the training data each time (which the user can change using random1Dict and random2Dict in "parameters_train_and_predict.py").

**predict.py**

- This Python file uses the model weights produced from the file "train.py" to predict outcomes using the prediction data. This produces the final results (please see below).

**functions.py**

- This Python file contains the main functions used to run the Python scripts.

**parameters_gridsearch.py**

- This Python file contains the variables and modelling parameters used for the grid search algoritm in "gridsearch.py".

**parameters_train_and_predict.py**

- This Python file contains the variables and modelling parameters used for the training algoritm in "train.py" and prediction algorithm in "predict.py".

## Sample results

**Classification models**
Model | Accuracy | F1 | Precision | Recall | ROC Area under the curve
--- | --- | --- | --- |--- |--- 
Random Forest | 87% | 82% | 96% | 72% | 85%
XGBoost | 86% | 81% | 94% | 72% | 84%
Logistic Regression | 82% | 75% | 92% | 63% | 80%
Naive-Bayes | 79% | 71% | 83% | 63% | 77%
Support-Vector Machines | 77% | 69% | 79% | 61% | 74%
Light GBM | 73% | 52% | 97% | 34% | 67%



**Regression models**
Model | Mean Absolute Error | Mean Squared Error | Predicted Obs. within 10% | Predicted Obs. within 20% | Predicted Obs. within 30% | Predicted Obs. 30%+ |Predicted % Obs. within 10% | Predicted % Obs. within 20% | Predicted % Obs. within 30% | Predicted %Obs. 30%+
--- | --- | --- | --- |--- |--- | --- |--- |--- | --- |--- 
Support-Vector Machines | 96265 | 70713063770 | 6486 | 6134 | 5857 | 47058 | 10% | 9% | 9% | 72%
Random forest | 159348 | 101936266961 | 3181 | 3112 | 3221 | 56021 | 5% | 5% | 5% | 85%
Light GBM | 161543 | 110411410042 | 3488 | 3500 | 3461 | 55086 | 5% | 5% | 5% | 84%
XGBoost | 162485 | 107937880996 | 3665 | 3612 | 3598 | 54660 | 6% | 6% | 5% | 83%
Bayes Ridge | 167638 | 115101599398 | 3087 | 3121 | 3075 | 56252 | 5% | 5% | 5% | 86%
Linear Regression | 167988 | 115611556298 | 3068 | 3131 | 3012 | 56324 | 5% | 5% | 5% | 86%

## Sample feature importance charts (top 30 variables)

![alt text](https://github.com/rethamlai/customer-prediction-using-machine-learning/blob/main/Charts/20210325_LogisticRegression_feat_imp.png?raw=true)

![alt text](https://github.com/rethamlai/customer-prediction-using-machine-learning/blob/main/Charts/20210325_XGBRegressor_feat_imp.png?raw=true)
