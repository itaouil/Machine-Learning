"""
    Support Vector Regression
"""

# Modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import dataset and build matrix of features
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
Y = dataset.iloc[:, 2].values

# Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_y.fit_transform(Y)

# SVR fitting
from sklearn.svm import SVR
regressor = SVR(kernel='rbf') # non-linear kernel, of course :)
regressor.fit(X, Y)

# Prediction
Y_pred = sc_Y.inverse_transform(regressor.predict(sc_X.fit_transform(np.array([[6.5]]))))

# Plot SVR results
plt.scatter(X, Y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Lever')
plt.ylabel('Salary')
plt.show()
