# importing all the libraries
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from training import LRModel

# Building a Predictive System


inputData = (13.54, 14.36, 87.46, 566.3, 0.09779, 0.08129, 0.06664, 0.04781, 0.1885, 0.05766, 0.2699, 0.7886, 2.058, 23.56,
             0.008462, 0.0146, 0.02387, 0.01315, 0.0198, 0.0023, 15.11, 19.26, 99.7, 711.2, 0.144, 0.1773, 0.239, 0.1288, 0.2977, 0.07259)
# change input data to a numpy array
inputDataArray = np.asarray(inputData)
# reshape the numpy array as we are predicting one datapoint
inputDataReshaped = inputDataArray.reshape(1, -1)

prediction = LRModel.predict(inputDataReshaped)
if prediction[0] == 0:
    print("The Breast Cancer is Malignant.")
else:
    print("The Breast Cancer is Benign.")
