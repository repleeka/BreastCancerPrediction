# importing all the libraries
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from preprocessing import *
from training import *

# Model Evaluation
#  - Accuracy Score

# Accuracy on training data
XtrainPrediction = LRModel.predict(X_train)
training_data_accuracy = accuracy_score(Y_train, XtrainPrediction)
print("Accuracy on training data = {}".format(training_data_accuracy))

# Accuracy on test data
XtestPrediction = LRModel.predict(X_test)
test_data_accuracy = accuracy_score(Y_test, XtestPrediction)
print("Accuracy on test data = {}".format(test_data_accuracy))
