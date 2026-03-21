import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.linear_model import LinearRegression as lr
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
import csv
from sklearn.model_selection import KFold, cross_val_score
from sklearn.svm import SVC

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


def fillempty(data):
    filled_data = data
    filled_data["A1Cresult"] = data['A1Cresult'].fillna('none')
    filled_data["max_glu_serum"] = data['max_glu_serum'].fillna('none')
    filled_data["race"] = data["race"].replace('?', 'Caucasian') #put together NaNs with Caucasian? most common (only 2% of values were missing)
    return filled_data


def drop_columns(data):
    return data.drop(columns=[
        "weight", "payer_code", "medical_specialty",
        "encounter_id", "examide", "troglitazone",
        "citoglipton", "diag_1", "diag_2", "diag_3"
    ])

def ids_mapping(data):
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
    fixed_data = data
    fixed_data['admission_type'] = apply_grouped_mapping(data, 'admission_type_id', admission_type_groups)
    fixed_data['discharge_disposition'] = apply_grouped_mapping(data, 'discharge_disposition_id', discharge_groups)
    fixed_data['admission_source'] = apply_grouped_mapping(data, 'admission_source_id', admission_source_groups)
    return fixed_data


def map_data(data):
    mapped_data = data
    mapped_data["age"] = data["age"].map({
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
    mapped_data["tolazamide"] = data["tolazamide"].map({
        'No':0,
        'Steady':1,
        'Up':2
    })
    #Binary ---------------------------------------------------------------------
    mapped_data["acetohexamide"] = data["acetohexamide"].map({
        'No':0,
        'Steady':1
    })
    mapped_data["glimepiride-pioglitazone"] = data["glimepiride-pioglitazone"].map({
        'No':0,
        'Steady':1
    })
    mapped_data["metformin-pioglitazone"] = data["metformin-pioglitazone"].map({
        'No':0,
        'Steady':1
    })
    mapped_data["metformin-rosiglitazone"] = data["metformin-rosiglitazone"].map({
        'No':0,
        'Steady':1
    })
    mapped_data["glipizide-metformin"] = data["glipizide-metformin"].map({
        'No':0,
        'Steady':1
    })

    mapped_data["tolbutamide"] = data["tolbutamide"].map({
        'No':0,
        'Steady':1
    })
    mapped_data["change"] = data["change"].map({
        'Ch':1,
        'No':0
    })
    mapped_data["diabetesMed"] = data["diabetesMed"].map({
        'Yes':1,
        'No':0
    })
    return mapped_data

def onehot(data):
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
    # prefix
    fixed_data = pd.get_dummies(data, columns=nominal_cols, drop_first=True)
    return fixed_data

def fix_label(data):
    mapped_data = data
    mapped_data["readmitted"] = data["readmitted"].map({
            'No': 0,
            '<30': 1,
            '>30': 2
        })
    return mapped_data

def apply_preprocessing(data):
    """takes diabetes data set and preprocesses it"""
    fixed_data = fillempty(data)
    fixed_data = drop_columns(fixed_data)
    fixed_data = ids_mapping(fixed_data)
    fixed_data = map_data(fixed_data)
    fixed_data = onehot(fixed_data)

    return fixed_data


# -------------END OF FUNCTION
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
    # time_in_hospital 
    # num_lab_procedures 
    # num_procedures 
    # num_medications 
    # number_outpatient
    # number_emergency
    # number_inpatient 
    # diag_1 STR 685st      str(V57) REMOVED
    # diag_2 STR 692st      REMOVED 
    # diag_3 STR 746st      REMOVED
    # number_diagnoses 

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



# #--------------------------------------------------------------------------------
# #1.4 PCA 
# #https://www.geeksforgeeks.org/machine-learning/implementing-pca-in-python-with-scikit-learn/
# # nan_features, string_features = checkdata(data)
# # print(f"features that still have str: {string_features}")
# X = data.drop('readmitted', axis = 1)
# y = data['readmitted']

# # scaler = StandardScaler()
# # X_scaled = scaler.fit_transform(X)

# # pca = PCA(n_components=50)
# # X_pca = pca.fit_transform(X_scaled)


# #-------------------------------------------------------------------------------
# #1.5 Send to new file! FIXED DIABETES should be ready to train ml algorithms on!
# csv_file_path = 'fixed_diabetes.csv'

# data.to_csv(csv_file_path, index=False)

# print(f'CSV file &quot;{csv_file_path}&quot; has been created successfully.')


# ================================
# LOAD DATA
# ================================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# Save target separately
y = train["readmitted"]

# ================================
# PREPROCESS
# ================================
train_processed = apply_preprocessing(train)
test_processed = apply_preprocessing(test)

# Remove target from train features
train_processed = train_processed.drop(columns="readmitted")

# ================================
# ALIGN COLUMNS (CRITICAL)
# ================================
test_processed = test_processed.reindex(columns=train_processed.columns, fill_value=0)

# ================================
# SAVE FILES
# ================================
train_processed["readmitted"] = y  # add back target

train_processed.to_csv("train_processed.csv")
test_processed.to_csv("test_processed.csv")

print("✅ Files created:")
print("- train_processed.csv")
print("- test_processed.csv")


