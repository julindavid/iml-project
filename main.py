import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
import project


# COLORS!
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
    list1 = []
    list2 = []
    numbers = []
    for feature in data.columns:
        isnull = data[feature].isnull().sum()
        if isnull > 0:
            numbers.append(isnull)
            list1.append(feature)
            print(f"amount of null: {isnull}")

        if data[feature].dtype == 'object' or data[feature].dtype == 'string':
            list2.append(feature)
    return list1, list2


def map_label(data):
    data = data.copy()
    data["readmitted"] = data["readmitted"].map({
        'No': 0,
        '<30': 1,
        '>30': 2
    })
    return data


train_data = pd.read_csv('train_processed.csv')
test_data = pd.read_csv('test_processed.csv')
original_test = pd.read_csv("test.csv")
ids = original_test["id"]
X_test_kaggle = test_data.drop(columns=["id"], errors="ignore")
train_data = map_label(train_data)

train_NaN, train_string = checkdata(train_data)
test_Nan, test_string = checkdata(test_data)

print(f"train--> \n Nan: {train_NaN},\n String:{train_string}")
print("\n-----------------------------------------------------")
print(f"test--> \n Nan: {test_Nan},\n String:{test_string}")

project.print_uniq_vals2(train_data, "readmitted")

# -------------------------
# train model
# -------------------------
y = train_data["readmitted"]
X = train_data.drop(columns="readmitted")

# Remove id from features if it exists
X = X.drop(columns=["id"], errors="ignore")
X_test_kaggle = X_test_kaggle.reindex(columns=X.columns, fill_value=0)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=100,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {acc:.4f}")
print(classification_report(y_test, y_pred))

# Optional: Cross-Entropy / Log Loss on your validation split
y_proba = rf.predict_proba(X_test)
ce_loss = log_loss(y_test, y_proba, labels=[0, 1, 2])
print(f"Cross-Entropy (Log Loss): {ce_loss:.4f}")


# Load sample submission (THIS FIXES EVERYTHING)
sample = pd.read_csv("sample_submission.csv")

# Prepare test features (same as training)
X_test_kaggle = test_data.drop(columns=["id"], errors="ignore")

# Make sure columns match training EXACTLY
X_test_kaggle = X_test_kaggle.reindex(columns=X.columns, fill_value=0)

# Predict
kaggle_pred = rf.predict(X_test_kaggle)

# Convert labels back to strings
reverse_map = {0: 'No', 1: '<30', 2: '>30'}
kaggle_pred = pd.Series(kaggle_pred).map(reverse_map)

# Save submission
submission = pd.DataFrame({
    "id": ids,
    "readmitted": kaggle_pred
})

submission.to_csv("kaggle_result.csv", index=False)

print("✅ submission.csv created correctly")