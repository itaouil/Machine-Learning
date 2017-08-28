# **Data Preprocessing*

The following section serves as a description of the possible steps regarding data preprocessing preceding steps in a Machine Learning problem, before the model creation.

## Import libraries

Well it seems obvious, but of course we need some setup work, such as importing the needed libraries to interact with our dataset, plot visual content like graphs and perform some computational work. If Python is your language of choice (as in my case) these are most probably the libraries you want to start with:

- **Numpy** (math/statistical library)
- **Matplotlib** (graph plotting)
- **Pandas** (manage datasets)

Piece of code:

```Python
# Import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

## Import dataset

The next step along the line is to import our dataset and construct the matrix of features englobing the dependent and independent variables.

Piece of code:

```Python
# Read dataset
dataset = pd.read_csv('Data.csv')

# Extract independent variables
X = dataset.iloc[, :-1].values

# Extract dependent variables
Y = dataset.iloc[, 3].values
```

## Handle missing data

In case of missing data in our dataset, we have few options. We can either decide to remove all the lines in the dataset with missing content, however, with this approach we might get rid of critical data. Hence, it is a better choice to substitute the missing value or values with one of the following outputs:

- Median of the feature column
- Mean of the feature column
- Most Frequent Value of the feature column

Piece of code:

```Python
# Import needed library
from sklearn.preprocessing import Imputer

# Create mean value based instance
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)

# Fit imputer instance to data
imputer = imputer.fit(X[:, 1:3])

# Transform data
X[:, 1:3] = imputer.transform(X[:, 1:3])
```

## Encode Categorical Data

Categorical data are non-numerical values in the datasets. ML, however, is based upon statistical and mathematical methods that make use of numbers, hence, the necessity to encode these categorical data into a number representation.

**P.S:** Avoid relational order in the encoding if the categorical values don't have any, so as not to fool the ML model (Use dummy variables instead).

```Python
# Import encoding library
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Encode independent variables
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Encode dependent variables
labelencoder_y = LabelEncoder()
Y = labelencoder_y.fit_transform(y)
```

## Dataset split

The next important step is to divide the dataset into two parts, a testing section and a training section. Why is that ? Because when we train our model on the training set we want to make sure the model did not learn content by heart, but instead it found the correlations between the Xs and the Ys, hence why we need to test it.

Piece of code:

```Python
# Import needed library
from sklearn.cross_validation import train_test_split

# Split dataset in a 80:20 ration
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 0)
```

## Feature scaling

Last but not least we might want to feature scale some of our features especially if we have big value differences between our independent features. This big differences might disrupt some ML models based on euclidean distances, making certain features redundant. **You could scale as well for faster convergence**.

There are two types of scaling:

- Standarisation scaling
- Normalisation scaling

Piecen of code:

```Python
# Import needed library
from sklearn.preprocessing import StandardScaler

# Scale independent variables
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Scale dependent variables
sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)
```
