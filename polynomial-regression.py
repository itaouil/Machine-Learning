# Polynomial regression baby

"""
    Data Preprocessing
"""

# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Create matrix features
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Split dataset (training and testing)
# from sklearn.model_selection import train_test_split
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

"""
    Linear regression model (benchmark)
"""

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, Y) # find correlations

"""
    Polynomial linear regression
"""

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, Y)

"""
    Fit polynomial regressor
    into multiple linear regressor.
"""

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, Y)

"""
    Visualise linear regression results
"""

plt.scatter(X, Y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth of Bluff [Linear Regression]')
plt.xlabel('Position level')
plt.ylabel('Salary')
# plt.show()

"""
    Visualise polynomial regression results
"""

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, Y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth of Bluff [Polynomial Regression]')
plt.xlabel('Position level')
plt.ylabel('Salary')
# plt.show()

"""
    Predict 6.5 position linear regression
"""

print(lin_reg.predict(6.5))

"""
    Predict 6.5 position polynomial regression
"""

print(lin_reg_2.predict(poly_reg.fit_transform(6.5)))
