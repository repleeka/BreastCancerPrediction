# importing all the libraries
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Model Training using Logistic Regression

# defining the model
LRModel = LogisticRegression()

# Training the logistic regression model using Training data
# Finding the relationship between the x-train and y-train values
model.fit(X_train, Y_train)
