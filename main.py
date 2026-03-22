import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss, confusion_matrix
from sklearn.decomposition import PCA
import project
import matplotlib.pyplot as plt
import seaborn as sns

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
print(train_data.columns)
features_to_drop = ['patient_nbr', 'Unnamed: 0']
train_data = train_data.drop(columns=features_to_drop, errors='ignore')
test_data = test_data.drop(columns=features_to_drop, errors='ignore')

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

#Model
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

sample = pd.read_csv("sample_submission.csv")

#Match kaggle columns
X_test_kaggle = test_data.drop(columns=["id", "unnamed: 0"], errors="ignore")
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

#Transfer result to csv file
submission.to_csv("kaggle_result.csv", index=False)

# FIX PLOTS AND RESULTS 

# Compute confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=[0,1,2])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=['No','<30','>30'], yticklabels=['No','<30','>30'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Feature importance
importances = rf.feature_importances_
feat_names = X.columns
feat_importance_df = pd.DataFrame({'Feature': feat_names, 'Importance': importances})
feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False).head(20)  # Top 20 features

plt.figure(figsize=(10,6))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df)
plt.title("Top 20 Feature Importances")
plt.show()



from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# Binarize the true labels for multi-class PR curve
y_test_binarized = label_binarize(y_test, classes=[0,1,2])
n_classes = y_test_binarized.shape[1]

# Get predicted probabilities
y_score = rf.predict_proba(X_test)

plt.figure(figsize=(8,6))

colors = ['blue', 'green', 'red']
class_names = ['No', '<30', '>30']

for i in range(n_classes):
    precision, recall, _ = precision_recall_curve(y_test_binarized[:, i], y_score[i][:,i] if isinstance(y_score, list) else y_score[:, i])
    ap = average_precision_score(y_test_binarized[:, i], y_score[i][:,i] if isinstance(y_score, list) else y_score[:, i])
    plt.plot(recall, precision, color=colors[i], lw=2,
             label=f'Class {class_names[i]} (AP={ap:.2f})')

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curves (Validation Set)')
plt.legend(loc='lower left')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()

