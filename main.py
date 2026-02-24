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


mapping = pd.read_csv("IDS_mapping.csv") # Given mapping
data = pd.read_csv("diabetic_data.csv") # All diabetic data


#DROPPED DATA That had not enough values to use OR had no value, or irrelevant data.

data = data.drop(columns="weight") #low amount of values
data = data.drop(columns="payer_code") #irrelevant
data = data.drop(columns="medical_specialty") #irrelevant
data = data.drop(columns="encounter_id") #irrelevant
data = data.drop(columns="patient_nbr") #irrelevant

#--------------------------------------------------------------------------------------

data["A1Cresult"] = data['A1Cresult'].fillna('none')
data["max_glu_serum"] = data['max_glu_serum'].fillna('none')

length = len(data.keys())
print(f"length of keys: {length}")
# print(f" after dropping columns {len(data)}"


def checkdata(data): # checks a data if there are any NaN values in any value.
    for feature in data.keys():
        test = data[feature].isnull().sum()
        
        print(f"Feature: {feature}, NaN values: {test}")


def check_missing_values(data):
    """Check and print missing values in the dataset."""
    nan_values_per_feature = data.isnull().sum()
    nan_total = sum(list(data.isnull().sum()))
    
    print(f"The dataset length: \t\t{c.BLUE}{len(data)}{c.END}")
    print(f"Total number of missing values: {c.BOLD}{nan_total}{c.END}\n")
    print(f"{c.BOLD}Printing how many entries in each column contain no NaN values{c.END}:")



# -------------------------------------------

target = data["readmitted"]
train = data.drop(columns="readmitted") #TODO: MÅSTE TOLKA READMITTED TILL NÅGOT ANNAT, får ej vara siffror
# KANSKE BERÄKNA PROBABILITIES, ex. Count NO / TOTAL count, eller One HOT





encoder = OneHotEncoder()
#race = encoder.fit_transform(data["race"])
#race_cols = encoder.get_feature_names_out(["race"])
#data[race_cols] = race
#data.drop(columns=["race"], inplace=True)


# One hot, divide race into different columns, that we put into a new DF (fixed_df)
race = pd.get_dummies(data["race"].apply(pd.Series).stack()).groupby(level=0).sum()
gender = pd.get_dummies(data["gender"].apply(pd.Series).stack()).groupby(level=0).sum()
age = pd.get_dummies(data["age"].apply(pd.Series).stack()).groupby(level=0).sum()
# = pd.get_dummies(data["race"].apply(pd.Series).stack()).groupby(level=0).sum()
fixed_df = data.drop(["race", "gender", "age"], axis=1).join(race).join(gender).join(age)
age = encoder.fit_transform(data[["age"]])
print(fixed_df["Asian"])

#Split the data into train & test
# x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=42)
# print(data["race"])

fixed_df.info()