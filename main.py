import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder


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

data = pd.read_csv("fixed_diabetic.csv") # FIXED diabetic data

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

l1, l2 = checkdata(data)

