from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np

data_frame = pd.read_csv("iris_data_with_class_label.csv", sep=",",
                         names=["sepal_length", "sepal_width", "petal_length",
                                "petal_width", "Class"])

CLASSES = data_frame.Class.unique()
print(CLASSES)

print(data_frame.shape)
print(data_frame.head())
print(data_frame.describe())