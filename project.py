import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

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
print(f"features that have NaN values: {nan_features}")
print(f"features that have string values: {string_features}")
train_data = train_data.drop(columns="encounter_id") #irrelevant
train_data = train_data.drop(columns="examide") # Only one value (NO)
train_data = train_data.drop(columns="troglitazone") # Only one value (NO)
train_data = train_data.drop(columns="citoglipton") # Only one value (NO)
# train_data = train_data.drop(columns="patient_nbr") #irrelevant

#----------------------------------------------------------------------------------------------------
# 1.2 Remap Str-Features into numbers
train_data["race"] = train_data["race"].replace('?', 'Other')

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
race_dummies = pd.get_dummies(train_data['race'], drop_first=True)
gender_dummies = pd.get_dummies(train_data['gender'])
diag1_dummies = pd.get_dummies(train_data['diag_1'], prefix='diag1')
diag2_dummies = pd.get_dummies(train_data['diag_2'], prefix='diag2')
diag3_dummies = pd.get_dummies(train_data['diag_3'], prefix='diag3')
metformin_dummies = pd.get_dummies(train_data['metformin'], prefix='metformin')
repaglinide_dummies = pd.get_dummies(train_data['repaglinide'], prefix='repaglinide')
nateglinide_dummies = pd.get_dummies(train_data['nateglinide'], prefix='nateglinide')
chlorpropamide_dummies = pd.get_dummies(train_data['chlorpropamide'], prefix='chlorpropamide')
glimepiride_dummies = pd.get_dummies(train_data['glimepiride'], prefix='glimepiride')
glipizide_dummies = pd.get_dummies(train_data['glipizide'], prefix='glipizide')
glyburide_dummies = pd.get_dummies(train_data['glyburide'], prefix='glyburide')
pioglitazone_dummies = pd.get_dummies(train_data['pioglitazone'], prefix='pioglitazone')
rosiglitazone_dummies = pd.get_dummies(train_data['rosiglitazone'], prefix='rosiglitazone')
acarbose_dummies = pd.get_dummies(train_data['acarbose'], prefix='acarbose')
miglitol_dummies = pd.get_dummies(train_data, columns=['miglitol'], prefix='miglitol')
insulin_dummies = pd.get_dummies(train_data['insulin'], prefix='insulin')
glyburide_dummies = pd.get_dummies(train_data['glyburide-metformin'], prefix='glyburide')
max_glu_serum_dummies = pd.get_dummies(train_data['max_glu_serum'], prefix='max_glu_serum')
A1Cresult_dummies = pd.get_dummies(train_data['A1Cresult'], prefix='A1Cresult')


#--------------------------------------------------------------------------------------------------------------
# 1.3 Handle IDS_Mapping file

# Load the mapping file
mapping = pd.read_csv('IDS_mapping.csv')


# admission_type_id IMPORTANT 9
#admission_type_id:
# (Since the file is stacked, you might want to manually extract sections or just refer to it)
# print(mapping.iloc[0:8])
train_data["discharge_disposition_id"] = train_data["discharge_disposition_id"].map({
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    7: 5,
    5: 6,
    6: 7, #Group together: NaN & Not Mapped ?
    8: 7
})

#Needs to be handled separate?
# discharge_disposition_id IMPORTANT 8
print(mapping.iloc[10:40])
# train_data["discharge_disposition_id"].map({
#     1: 1,#discharged to home,
#     2: 2, #Discharged/transferred to another short term
#     3: 3,
#     4: 4,
#     5: 5,
#     6: 6,
#     7: 7,
#     8:8,
#     9:9,
#     10:10,
#     11:11,
#     12:12,
#     13:13,
#     14:14,
#     15:15,
#     16:16,
#     17:17,
#     18:18, NaN
#     19:19, Expired
#     20:20, Expired
#     21:21, Expired


# })

# admission_source_id
# print(mapping.iloc[42:67])
#--------------------------------------------------------------------------------------------------------------
#1.4 Reduce Dimensions PCA

#2. Create models

#3. plot and compare results


# print_uniq_val(train_data, train_data.keys())