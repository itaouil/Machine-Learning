# **Simple Linear Regression**

The following are the steps and the explanation of the code found in the simple-linear-regression.py file.

The model we are trying to build is such that we can find the correlation between the expected salary and the number of years of experience when a new employee joins the company (business model).

Simple linear regression is a type of regression model that tries to find a linear relationship between two features, and ultimately find the line of best fit for the data points. Now, we can have many different lines defining this correlation, how do we find the best one ? We use a technique called **square sum**, and by finding the minimum sum (**least square error**) we are certain to have the best line fitting the data points.

## **Steps**

These are the steps described in the ML python script:

1. Preprocess data
 - Import libraries and dataset/s
 - Handle missing data (NOT needed)
 - Encode categorial features (NOT needed)
 - Split dataset into test and training set
 - Feature scaling (NOT needed as already part of the regressor)
2. Create regression model and fit the training set
3. Predict dependent variables with test set
4. Plot charts
