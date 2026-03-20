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

#Colors used in plots and other stuff!
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

def print_uniq_val(data, column_names):
    # Iterate directly over the names (no need for range(len))
    for col in column_names:
        print(f"{c.BOLD}{col}{c.END} unique values:")
        print(data[col].unique())
        print("-" * 30)

def print_uniq_vals2(data, column):
    # Iterate directly over the names (no need for range(len)
    
    print(data[column].unique())
    print("-" * 30)


def check_missing_values(data):
    """Check and print missing values in the dataset."""
    nan_values_per_feature = data.isnull().sum()
    nan_total = sum(list(data.isnull().sum()))
    
    print(f"The dataset length: \t\t{c.BLUE}{len(data)}{c.END}")
    print(f"Total number of missing values: {c.BOLD}{nan_total}{c.END}\n")
    print(f"{c.BOLD}Printing how many entries in each column contain no NaN values{c.END}:")


def apply_grouped_mapping(df, column, grouping_dict):
    """
    Flattens a grouping dictionary and maps it to a DataFrame column.
    """
    # Create the flat map: {1: 'Emergency', 7: 'Emergency', 2: 'Urgent'...}
    flat_map = {idx: group_name for group_name, id_list in grouping_dict.items() for idx in id_list}
    
    # Map the new names and fill any missing IDs with 'Other'
    return df[column].map(flat_map).fillna('Other')
data = pd.read_csv("train.csv")
keys = data.keys()

#helper functions, handle binary



# keys found:   -----------------
    # encounter_id REMOVED
    # weight REMOVED
    #examide REMOVED
    #payer code REMOVED
    #medical specialty REMOVED
    #troglitazone REMOVED
    #citoglipton REMOVED
    #diag_1
    #diag_2
    #diag_3


    # patient_nbr
    # time_in_hospital IMPORTANT 3
    # num_lab_procedures IMPORTANT 1
    # num_procedures IMPORTANT 5
    # num_medications IMPORTANT 2
    # number_outpatient
    # number_emergency
    # number_inpatient IMPORTANT 7
    # diag_1 STR 685st      str(V57)
    # diag_2 STR 692st      
    # diag_3 STR 746st      
    # number_diagnoses IMPORTANT 6

    #BINARY ------------
    # acetohexamide STR ['No', 'Steady'] HANDLED MAP
    # tolbutamide STR ['No', 'Steady'] HANDLED MAP
    # change STR ['Ch', 'No'] HANDLED MAP
    # diabetesMed STR ['Yes', 'No'] HANDLED MAP
    # glimepiride-pioglitazone STR ['No', 'Steady'] HANDLED MAP
    # metformin-rosiglitazone STR ['No', 'Steady'] HANDLED MAP
    # metformin-pioglitazone STR ['No', 'Steady'] HANDLED MAP
    # glipizide-metformin STR ['No', 'Steady'] #HANDLED MAP

    #ORDINAL ----------- 
    # age STR IMPORTANT 4 [ '[60-70)',  '[70-80)',  '[50-60)',  '[80-90)',  '[40-50)',  '[20-30)', '[30-40)',  '[10-20)', '[90-100)',   '[0-10)'] MAP HANDLED 
    # readmitted STR ['No', '>30', '<30']  MAP HANDLED
    

    #NOMINAL ---------- FIXED WITH ONE HOT!
    # race STR ['Caucasian', 'AfricanAmerican', 'Hispanic', 'Asian', 'Other', '?'] ONEHOT HANDLED
    # gender STR ['Male', 'Female', 'Unknown/Invalid'] ONEHOT HANDLED
    # metformin STR ['No', 'Steady', 'Up', 'Down'] ONEHOT HANDLED
    # repaglinide STR ['No', 'Steady', 'Down', 'Up'] ONEHOT HANDLED
    # nateglinide STR ['No', 'Steady', 'Down', 'Up'] ONEHOT HANDLED
    # chlorpropamide STR ['No', 'Steady', 'Up'] ONEHOT HANDLED
    # glimepiride STR ['No', 'Steady', 'Down', 'Up'] ONEHOT HANDLED
    # glipizide STR ['No', 'Steady', 'Up', 'Down'] ONEHOT HANDLED
    # glyburide STR ['Steady', 'No', 'Up', 'Down'] ONEHOT HANDLED
    #'pioglitazone STR ['No', 'Steady', 'Down', 'Up'] ONEHOT HANDLED
    # rosiglitazone STR ['No', 'Steady', 'Up', 'Down'] ONEHOT HANDLED
    # acarbose STR ['No', 'Steady', 'Down', 'Up'] ONEHOT HANDLED
    # miglitol STR ['No', 'Steady', 'Down', 'Up'] ONEHOT HANDLED
    # insulin STR IMPORTANT 10 ['Steady', 'No', 'Up', 'Down'] ONEHOT HANDLED
    # glyburide-metformin STR ['No', 'Steady', 'Up', 'Down'] ONESHOT HANDLED
    # max_glu_serum STR ['none', '>300', '>200', 'Norm'] ONESHOT HANDLED
    # A1Cresult STR [none, '>8', 'Norm', '>7'] ONESHOT HANDLED
    #race HANDLED
    #gender HANDLED


#https://www.geeksforgeeks.org/data-analysis/data-preprocessing-machine-learning-python/

#========================================================================
#1. Preprocessing of data 
#========================================================================
#1.1: Handle missing values from features: 
# print_uniq_vals2(data, "A1Cresult")

data["A1Cresult"] = data['A1Cresult'].fillna('none')
data["max_glu_serum"] = data['max_glu_serum'].fillna('none')
data["race"] = data["race"].replace('?', 'Caucasian') #put together NaNs with Caucasian? most common (only 2% of values were missing) 

#1.1 Remove features with a lot of missing values & unecessary features
data = data.drop(columns="weight") #low amount of values
data = data.drop(columns="payer_code") #low amount of values
data = data.drop(columns="medical_specialty") #low amount of values
data = data.drop(columns="encounter_id") #irrelevant
data = data.drop(columns="examide") # Only one value (NO)
data = data.drop(columns="troglitazone") # Only one value (NO)
data = data.drop(columns="citoglipton") # Only one value (NO)
data = data.drop(columns="diag_1") # String values, too high cardinality  example. 456 - 600 heart_disease
data = data.drop(columns="diag_2") # String values, too high cardinality
data = data.drop(columns="diag_3") # String values, too high cardinality

#--------------------------------------------------------------------------------------------------------------
# 1.2 Handle IDS_Mapping file

# Load the mapping file
mapping = pd.read_csv('IDS_mapping.csv')

admission_type_groups = {
    'Emergency': [1, 7],
    'Urgent': [2],
    'Elective': [3],
    'Newborn': [4],
    'Unknown': [5, 6, 8]
}

# Grouping for discharge_disposition_id
discharge_groups = {
    'Home': [1, 6, 8],
    'Expired': [11, 19, 20, 21],
    'Transferred': [2, 3, 4, 5, 10, 12, 13, 14, 15, 16, 17, 22, 23, 24, 27, 28, 29, 30],
    'Unknown': [18, 25, 26]
}

# Grouping for admission_source_id
admission_source_groups = {
    'Referral': [1, 2, 3],
    'Transfer': [4, 5, 6, 10, 18, 19, 22, 25, 26],
    'Emergency': [7],
    'Birth': [11, 12, 13, 14, 23, 24],
    'Unknown': [9, 15, 17, 20, 21],
    'Other': [8]
}

data['admission_type'] = apply_grouped_mapping(data, 'admission_type_id', admission_type_groups)
data['discharge_disposition'] = apply_grouped_mapping(data, 'discharge_disposition_id', discharge_groups)
data['admission_source'] = apply_grouped_mapping(data, 'admission_source_id', admission_source_groups)

#----------------------------------------------------------------------------------------------------
# 1.3 Remap Str-Features into numbers

data["age"] = data["age"].map({
    '[0-10)': 0,
    '[10-20)': 1,
    '[20-30)': 2,
    '[30-40)': 3,
    '[40-50)': 4,
    '[50-60)': 5,
    '[60-70)': 6,
    '[70-80)': 7,
    '[80-90)': 8,
    '[90-100)': 9})
data["readmitted"] = data["readmitted"].map({
    'No': 0,
    '<30': 1,
    '>30': 2
})
data["acetohexamide"] = data["acetohexamide"].map({
    'No':0,
    'Steady':1
})
data["glimepiride-pioglitazone"] = data["glimepiride-pioglitazone"].map({
    'No':0,
    'Steady':1
})
data["metformin-pioglitazone"] = data["metformin-pioglitazone"].map({
    'No':0,
    'Steady':1
})
data["metformin-rosiglitazone"] = data["metformin-rosiglitazone"].map({
    'No':0,
    'Steady':1
})
data["glipizide-metformin"] = data["glipizide-metformin"].map({
    'No':0,
    'Steady':1
})

data["tolbutamide"] = data["tolbutamide"].map({
    'No':0,
    'Steady':1
})
data["tolazamide"] = data["tolazamide"].map({
    'No':0,
    'Steady':1,
    'Up':2
})
data["change"] = data["change"].map({
    'Ch':1,
    'No':0
})
data["diabetesMed"] = data["diabetesMed"].map({
    'Yes':1,
    'No':0
})


#One-hot
encoder = OneHotEncoder()
# 1. Define all columns that should be turned into binary 0s and 1s
# Note: I removed the diag columns to prevent a crash—group them first!
nominal_cols = [
    'race', 'gender', 'metformin', 'repaglinide', 'nateglinide', 
    'chlorpropamide', 'glimepiride', 'glipizide', 'glyburide', 
    'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 
    'insulin', 'glyburide-metformin', 'max_glu_serum', 'A1Cresult',
    'admission_type', 'discharge_disposition', 'admission_source'
]

# 2. Run ONE command for the whole dataset
# This replaces all the lines in your snippet
data = pd.get_dummies(data, columns=nominal_cols, drop_first=True)

#--------------------------------------------------------------------------------
#1.4 PCA 
#https://www.geeksforgeeks.org/machine-learning/implementing-pca-in-python-with-scikit-learn/
# nan_features, string_features = checkdata(data)
# print(f"features that still have str: {string_features}")
X = data.drop('readmitted', axis = 1)
y = data['readmitted']
    
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# pca = PCA(n_components=50)
# X_pca = pca.fit_transform(X_scaled)

# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=13)

# model = LogisticRegression(max_iter=1000, class_weight='balanced')
model = RandomForestClassifier(n_estimators=500, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(y_pred.shape[0])
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',     xticklabels=['No', '<30', '>30'], yticklabels=['No', '<30', '>30'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Random Forest)')
plt.show()


#HÄR ÄR KODEN SOM FELIX HAR SOM BESKRIVER -----------
#print("PCA börjar nu")
#X_scaled = StandardScaler().fit_transform(X_train)
#pca = PCA(n_components=10)
#X_transformed = pca.fit_transform(X_scaled)
#eigenvalues = pca.explained_variance_
#
#plt.plot(eigenvalues)
#plt.show()

#-------------------------------------------------------------------------------
#1.5 Send to new file! FIXED DIABETES should be ready to train ml algorithms on!
csv_file_path = 'fixed_diabetes.csv'

# data.to_csv(csv_file_path, index=False)

# print(f'CSV file &quot;{csv_file_path}&quot; has been created successfully.')


