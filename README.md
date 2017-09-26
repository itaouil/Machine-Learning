# **Random Forest Regression**

As the name of this regression model portraits, we are talking about many decision regression trees.

### How does it work ?

Here is what the model comports.

1. We choose a subset of the dataset and train a decision tree on it.
2. We chose another subset of the dataset and train another decision tree on it.
3. Keep repeating 1 and 2 until the whole dataset is used.
4. When an unseen datapoint is presented we predict the dependent value on all the decision tree we modeled before and the value of the prediction is the average of all the prediction.

**P.S:**: What we obtain is a non-linear, non-continuos model that is a bit less subtle to outliers and statistically speaking performs better that a single decision tree regression model.

![alt](https://github.com/itaouil/Machine-Learning/blob/07-random-forest-regression/rfr.png)
