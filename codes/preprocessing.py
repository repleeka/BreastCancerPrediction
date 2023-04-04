# importing all the libraries
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data collection and Preprocessing


# loading the dataset from sklearn
breastCancerDataset = sklearn.datasets.load_breast_cancer()

# loading data to a dataframe
dataFrame = pd.DataFrame(breastCancerDataset.data,
                         columns=breastCancerDataset.feature_names)

# adding the 'target' column to the dataframe
dataFrame['label'] = breastCancerDataset.target

# exporting the dataset as a csv file
csvData = dataFrame.to_csv('../dataset/breastCancer.csv', index=False)

# number of rows and columns in the dataset
print("{}".format(dataFrame.shape))

# getting information about the data
dataFrame.info()

# checking for missing values
print(dataFrame.isnull().sum())

# statistical measures about the dataset
print(dataFrame.describe())

# checking the distribution of target variables
# 0 - Malignant
# 1 - Benign
print(dataFrame['label'].value_counts())
print(dataFrame.groupby('label').mean())
