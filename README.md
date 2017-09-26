# **Decision Tree Regression**

Decision trees are the first model we encounter in this machine-learning repo which is a non-linear and non-continuos. We are used already to non-linearity as we have encountered polynomial and svr type of regressions.

### What is this tree based regression model ?

It is a clustering model based on grouping the different independent variable based on information entropy. The regressor in sklearn offers different approaches, the one used in this case was the default minimum distance entropy, hence, grouping the different variables onto clusters and assigning to each of these leaves the average of the values contained within, aka our predicted values for the range.

However, we have to notice that performance wise polynomial regression did perform better. The result of the Decision Tree Model is the following:

![alt]()
