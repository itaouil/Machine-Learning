"""

    Simple linear regression model.

    The result is to create a model that outputs the salary of new employee in
    a company based on his/her years of
    experience and on the business model
    adopted by the company.

"""

""" Data preprocessing """

# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Read dataset
dataset = pd.read_csv('Salary_Data.csv')

# Create matrix features
X = dataset.iloc[:, :-1]
Y = dataset.iloc[:, 1]

# Split dataset (training and test set)
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

""" Simple Linear Regression fitting"""

# Fitting simple linear regression to training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

""" Prediction """

# Create a vector of predicted values
Y_pred = regressor.predict(X_test)

""" Plotting """

#Visualise the training set results
plt.scatter(X_train, Y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualise the test set results
plt.scatter(X_test, Y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
