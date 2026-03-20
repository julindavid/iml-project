import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import pandas as pd
import numpy as np
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


df = pd.read_csv('fixed_diabetes.csv')
df.head()


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

# l1, l2 = checkdata(data)

import project

# -------------------------
train = pd.read_csv("train.csv") # FIXED diabetic data

y = train['readmitted']

train_fixed = project.apply_preprocessing(train)
train_fixed = train_fixed.drop(columns="readmitted")

test = pd.read_csv("test.csv") # FIXED diabetic data
test_fixed = project.apply_preprocessing(test)

l1, l2 = checkdata(test_fixed)

print(f"Nan values check: {l1}")
print(f"string values check: {l2}")

# ------------------------------------------------------------
# TRAIN AND TEST!

X_train, X_test, y_train, y_test = train_test_split(train_fixed, y, test_size=0.2, random_state=13)
# model = LogisticRegression(max_iter=1000, class_weight='balanced')
model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

model.fit(train_fixed, y)
kaggle_pred = model.predict(test_fixed)

acc = accuracy_score(y_test, y_pred)
# print(y_pred.shape[0])
# cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {acc}")

print(y_pred.shape[0])
print(" \n\n")
print(y_pred)
DF = pd.DataFrame(kaggle_pred)
csv_file_path = 'kaggle_result.csv'
DF.to_csv(csv_file_path)
print(f'CSV file &quot;{csv_file_path}&quot; has been created successfully.')

# plt.figure(figsize=(5,4))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',     xticklabels=['No', '<30', '>30'], yticklabels=['No', '<30', '>30'])
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix (Random Forest)')
# plt.show()