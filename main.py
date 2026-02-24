import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression as lr
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


data = pd.read_csv("diabetic_data.csv")
target = data["readmitted"]
train = data.drop(columns="readmitted") #TODO: MÅSTE TOLKA READMITTED TILL NÅGOT ANNAT, får ej vara siffror
#KANSKE BERÄKNA PROBABILITIES, ex. Count NO / TOTAL count
# print(train.keys())

#Check if data has missing values -----------
nan_values_per_feature = data.isnull().sum()
nan_total = sum(list(data.isnull().sum()))

print(f"The dataset length: \t\t{c.BLUE}{len(data)}{c.END}")
print(f"Total number of missing values: {c.BOLD}{nan_total}{c.END}\n")

print(f"{c.BOLD}Printing how many entries in each column contain no NaN values{c.END}:")
data.info()




#Split the data into train & test
x_train, x_test, y_train, y_test = train_test_split(train, target, test_size=0.2, random_state=42)

lr1 = lr()
lr1.fit(x_train, y_train)
predict = lr1.predict(x_test)

plt.scatter(y_test, predict)
plt.xlabel("target data")
plt.ylabel("prediction data")
plt.legend()
plt.show()