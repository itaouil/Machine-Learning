# Multiple Linear Regression

""" Data preprocessing """

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing dataset and create matrix of features
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 4].values

# Encode categorial values (state feature)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoid Dummy Variable Trap
X = X[:, 1:]

# Split dataset (training and test sets)
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/5, random_state = 0)

""" All In """

# Fit class to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = regressor.predict(X_test)

""" Backward Elimination """

# Imports
import statsmodels.formula.api as sm

# Introduce b0 coefficients (add it to the equation)
X = np.append(arr = np.ones((50,1)).astype(int), values = X, axis = 1)


# Optimal statistical features (all of them at first)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# Fit new model (STEP 2)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
# P value checker (STEP 3)
# print(regressor_OLS.summary())

# Optimal statistical features
X_opt = X[:, [0, 1, 3, 4, 5]]
# Fit new model (STEP 2)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
# P value checker (STEP 3)
# print(regressor_OLS.summary())

# Optimal statistical features
X_opt = X[:, [0, 3, 4, 5]]
# Fit new model (STEP 2)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
# P value checker (STEP 3)
# print(regressor_OLS.summary())

# Optimal statistical features
X_opt = X[:, [0, 3, 5]]
# Fit new model (STEP 2)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
# P value checker (STEP 3)
# print(regressor_OLS.summary())

# Optimal statistical features
X_opt = X[:, [0, 3]]
# Fit new model (STEP 2)
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
# P value checker (STEP 3)
print(regressor_OLS.summary())

# Predicting the test set results
Y_pred_OLS = regressor_OLS.predict(X_test)
print(Y_test)
print(Y_pred_OLS)
print(Y_pred)
