# importing all the libraries
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Separating the features and target

# reading the dataset
dataframe = pd.read_csv('../dataset/breastCancer.csv')
print(dataframe.head())

# removing the label column and storing all the other values in
X = dataframe.drop(columns='label', axis=1)
# y contains all the label values
Y = dataframe['label']

# Spllitting the data into training and testing data
# four numpy arrays
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)
# test_size = 0.2 represents 20% of the data will be our test data and
# the remaining 80% will be our training data
# random_state = 2 represents the way in which the data will be splitted
print(X.shape, X_train.shape, X_test.shape)

# Model Training using Logistic Regression

# defining the model
LRModel = LogisticRegression()

# Training the logistic regression model using Training data
# Finding the relationship between the x-train and y-train values
LRModel.fit(X_train, Y_train)
