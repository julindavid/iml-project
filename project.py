import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import csv

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
train_data = pd.read_csv("train.csv")
keys = train_data.keys()

#helper functions, handle binary



# keys found:   -----------------
    # id
    # encounter_id REMOVED
    # patient_nbr
    # time_in_hospital IMPORTANT 3
    # num_lab_procedures IMPORTANT 1
    # num_procedures IMPORTANT 5
    # num_medications IMPORTANT 2
    # number_outpatient
    # number_emergency
    # number_inpatient IMPORTANT 7
    # diag_1 STR 685st      ONE HOT HANDLED
    # diag_2 STR 692st      ONE HOT HANDLED
    # diag_3 STR 746st      ONE HOT HANDLED
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
train_data["A1Cresult"] = train_data['A1Cresult'].fillna('none')
train_data["max_glu_serum"] = train_data['max_glu_serum'].fillna('none')
train_data["diag_3"] = train_data["diag_3"].fillna(0)

#1.1 Remove features with a lot of missing values & unecessary features
train_data = train_data.drop(columns="weight") #low amount of values
train_data = train_data.drop(columns="payer_code") #low amount of values
train_data = train_data.drop(columns="medical_specialty") #low amount of values


nan_features, string_features = checkdata(train_data)
# print(f"features that have NaN values: {nan_features}")
# print(f"features that have string values: {string_features}")
train_data = train_data.drop(columns="encounter_id") #irrelevant
train_data = train_data.drop(columns="examide") # Only one value (NO)
train_data = train_data.drop(columns="troglitazone") # Only one value (NO)
train_data = train_data.drop(columns="citoglipton") # Only one value (NO)

#--------------------------------------------------------------------------------------------------------------
# 1.2 Handle IDS_Mapping file

# Load the mapping file
mapping = pd.read_csv('IDS_mapping.csv')

# admission_type_id IMPORTANT 9
#admission_type_id:
# (Since the file is stacked, you might want to manually extract sections or just refer to it)
# print(mapping.iloc[0:8])

#Needs to be handled separate?
# discharge_disposition_id IMPORTANT 8
# print(mapping.iloc[10:40])

# Define your groupings in a clear dictionary
# Grouping for admission_type_id
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


train_data['admission_type'] = apply_grouped_mapping(train_data, 'admission_type_id', admission_type_groups)
train_data['discharge_disposition'] = apply_grouped_mapping(train_data, 'discharge_disposition_id', discharge_groups)
train_data['admission_source'] = apply_grouped_mapping(train_data, 'admission_source_id', admission_source_groups)

#----------------------------------------------------------------------------------------------------
# 1.3 Remap Str-Features into numbers
train_data["race"] = train_data["race"].replace('?', 'Caucasian') #put together NaNs with Caucasian? most common (only 2% of values were missing) 

train_data["age"] = train_data["age"].map({
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
train_data["readmitted"] = train_data["readmitted"].map({
    'No': 0,
    '<30': 1,
    '>30': 2
})
train_data["acetohexamide"] = train_data["acetohexamide"].map({
    'No':0,
    'Steady':1
})
train_data["glimepiride-pioglitazone"] = train_data["glimepiride-pioglitazone"].map({
    'No':0,
    'Steady':1
})
train_data["metformin-pioglitazone"] = train_data["metformin-pioglitazone"].map({
    'No':0,
    'Steady':1
})
train_data["metformin-rosiglitazone"] = train_data["metformin-rosiglitazone"].map({
    'No':0,
    'Steady':1
})
train_data["tolbutamide"] = train_data["tolbutamide"].map({
    'No':0,
    'Steady':1
})
train_data["change"] = train_data["change"].map({
    'Ch':1,
    'No':0
})
train_data["diabetesMed"] = train_data["diabetesMed"].map({
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
train_data = pd.get_dummies(train_data, columns=nominal_cols, drop_first=True)

csv_file_path = 'fixed_diabetes.csv'

train_data.to_csv(csv_file_path, index=False)

print(f'CSV file &quot;{csv_file_path}&quot; has been created successfully.')


#--------------------------------------------------------------------------------
#1.4 PCA

