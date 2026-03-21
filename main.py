import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import csv

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import project

train_data = pd.read_csv('train_processed.csv')
test_data = pd.read_csv('test_processed.csv')

#COLORS!
class c:
    PURPLE = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


#PREPROCESSING FIXED IN: project.py


def checkdata(data): 
    """
    checks a data if there are any null values in any feature.
    which features have string values that needs to be fixed
    returns list1 with all feature names that has NaN values & list2 of features with string values
    """
    list1 = [] # Features with NaN values
    list2 = [] # Features with string/object values
    numbers = []
    for feature in data.columns:
        # 1. Check for null values
        isnull = data[feature].isnull().sum()
        if isnull > 0:
            numbers.append(isnull)  
            list1.append(feature)
            print(f"amount of null: {isnull}")
            
        # 2. Check for string (object) values that need fixing/encoding
        # In pandas, strings are usually stored as 'object' or 'string'
        if data[feature].dtype == 'object' or data[feature].dtype == 'string':
            list2.append(feature)
    return list1, list2
def map_label(data):
    data["readmitted"] = data["readmitted"].map({
        'No': 0,
        '<30': 1,
        '>30': 2
    })
    return data

# test_data = project.fix_label(test_data)
train_data = map_label(train_data)

train_NaN, train_string = checkdata(train_data)
test_Nan, test_string = checkdata(test_data)

print(f"train--> \n Nan: {train_NaN},\n String:{train_string}")
print("\n-----------------------------------------------------")
print(f"test--> \n Nan: {test_Nan},\n String:{test_string}")

project.print_uniq_vals2(train_data, "readmitted")

#when submitting REVERSE result
reverse_map = {0: 'No', 1: '<30', 2: '>30'}
# submission["readmitted"] = submission["readmitted"].map(reverse_map)

#------------------------------------------------------------------------------
#train model