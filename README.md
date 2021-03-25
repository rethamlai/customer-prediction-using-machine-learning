# customer-prediction-using-machine-learning
An algorithm designed to take in customer level data for predicting future outcomes using different classification and regression methods

# Sample results

Model | Accuracy | F1 | Precision | Recall | ROC Area under the curve
--- | --- | --- | --- |--- |--- 
Random Forest | 87% | 82% | 96% | 72% | 85%
XGBoost | 86% | 81% | 94% | 72% | 84%
Logistic Regression | 82% | 75% | 92% | 63% | 80%
Naive-Bayes | 79% | 71% | 83% | 63% | 77%
Support Vector Machine | 77% | 69% | 79% | 61% | 74%
Light GBM | 73% | 52% | 97% | 34% | 67%


Model | Mean Absolute Error | Mean Squared Error | Predicted Obs. within 10% | Predicted Obs. within 20% | Predicted Obs. within 30% | Predicted Obs. 30%+ |Predicted % Obs. within 10% | Predicted % Obs. within 20% | Predicted % Obs. within 30% | Predicted %Obs. 30%+
--- | --- | --- | --- |--- |--- | --- |--- |--- | --- |--- 
Support Vector Machine | 96265 | 70713063770 | 6486 | 6134 | 5857 | 47058 | 10% | 9% | 9% | 72%
Random forest | 159348 | 101936266961 | 3181 | 3112 | 3221 | 56021 | 5% | 5% | 5% | 85%
Light GBM | 161543 | 110411410042 | 3488 | 3500 | 3461 | 55086 | 5% | 5% | 5% | 84%
XGBoost | 162485 | 107937880996 | 3665 | 3612 | 3598 | 54660 | 6% | 6% | 5% | 83%
Bayes Ridge | 167638 | 115101599398 | 3087 | 3121 | 3075 | 56252 | 5% | 5% | 5% | 86%
Linear Regression | 167988 | 115611556298 | 3068 | 3131 | 3012 | 56324 | 5% | 5% | 5% | 86%


