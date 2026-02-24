import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr
import seaborn as sns


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

nan_values_per_feature = data.isnull().sum()
nan_total = sum(list(data.isnull().sum()))

print(f"The dataset length: \t\t{c.BLUE}{len(data)}{c.END}")
print(f"Total number of missing values: {c.BOLD}{nan_total}{c.END}\n")

#DROPPED DATA That had not enough values to use OR had no value, or irrelevant data.
data = data.drop(columns="weight")
data = data.drop(columns="payer_code")
data = data.drop(columns="medical_specialty")

# data = data.dropna() # FIXME: 
length = len(data.keys())
print(f"length of keys: {length}")
# print(f" after dropping columns {len(data)}"

#Check if data has missing values -----------

nan_values_per_feature = data.isnull().sum()
nan_total = sum(list(data.isnull().sum()))

for feature in data.keys():
    test = data[feature].isnull().sum()
    
    # print(f"Feature: {feature}, NaN values: {test}")

# print(f"The dataset length: \t\t{c.BLUE}{len(data)}{c.END}")
# print(f"Total number of missing values: {c.BOLD}{nan_total}{c.END}\n")

# print(f"{c.BOLD}Printing how many entries in each column contain no NaN values{c.END}:")


# -------------------------------------------

target = data["readmitted"]
# train = data.drop(columns="readmitted") #TODO: MÅSTE TOLKA READMITTED TILL NÅGOT ANNAT, får ej vara siffror
# KANSKE BERÄKNA PROBABILITIES, ex. Count NO / TOTAL count



#Split the data into train & test
#x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=42)


print(data["max_glu_serum"].unique())