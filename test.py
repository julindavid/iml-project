import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)

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

# ============================================================
# 1. LOAD DATA
# ============================================================
data = pd.read_csv("diabetic_data.csv")
print(f"{c.BOLD}Raw dataset:{c.END} {data.shape[0]} rows, {data.shape[1]} columns\n")

# ============================================================
# 2. PREPROCESSING
# ============================================================

# 2a. Replace '?' with NaN everywhere
data.replace("?", np.nan, inplace=True)

# 2b. Drop columns with too many missing values or no predictive value
drop_cols = ["encounter_id", "patient_nbr", "weight", "payer_code", "medical_specialty"]
data.drop(columns=drop_cols, inplace=True)

# 2c. Drop high-cardinality diagnosis columns
data.drop(columns=["diag_1", "diag_2", "diag_3"], inplace=True)

# 2d. Remove patients who died or went to hospice (can't be readmitted)
expired_ids = [11, 13, 14, 19, 20]
data = data[~data["discharge_disposition_id"].isin(expired_ids)]
print(f"After removing expired/hospice patients: {len(data)} rows")

# 2e. Impute remaining NaN with column mode
for col in data.columns:
    if data[col].isnull().any():
        mode_val = data[col].mode()[0]
        data[col] = data[col].fillna(mode_val)

nan_total = data.isnull().sum().sum()
print(f"Remaining NaN values: {nan_total}")

# 2f. Encode age brackets as ordinal integers
age_map = {
    "[0-10)": 0, "[10-20)": 1, "[20-30)": 2, "[30-40)": 3, "[40-50)": 4,
    "[50-60)": 5, "[60-70)": 6, "[70-80)": 7, "[80-90)": 8, "[90-100)": 9,
}
data["age"] = data["age"].map(age_map)

# 2g. Encode medication columns as ordinal
med_map = {"No": 0, "Steady": 1, "Down": 2, "Up": 3}
med_cols = [
    "metformin", "repaglinide", "glimepiride", "glipizide",
    "glyburide", "pioglitazone", "rosiglitazone", "insulin",
    "glyburide-metformin",
]
for col in med_cols:
    data[col] = data[col].map(med_map)

# Drop near-zero-variance medication columns (>99% single value)
near_zero_var_meds = [
    "examide", "citoglipton", "acetohexamide", "troglitazone",
    "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone", "glipizide-metformin",
    "tolbutamide", "miglitol", "tolazamide", "chlorpropamide",
    "nateglinide", "acarbose",
]
data.drop(columns=near_zero_var_meds, inplace=True)

# 2h. Binary-encode simple binary columns
data["change"] = data["change"].map({"No": 0, "Ch": 1})
data["diabetesMed"] = data["diabetesMed"].map({"No": 0, "Yes": 1})

# 2i. One-hot encode remaining nominal features
nominal_cols = ["race", "gender", "max_glu_serum", "A1Cresult"]
data = pd.get_dummies(data, columns=nominal_cols, drop_first=True)

# 2j. Encode target variable
target_map = {"NO": 0, ">30": 1, "<30": 2}
data["readmitted"] = data["readmitted"].map(target_map)

print(f"\n{c.BOLD}Preprocessed dataset:{c.END} {data.shape[0]} rows, {data.shape[1] - 1} features")
print(f"\nTarget distribution:")
print(data["readmitted"].value_counts().rename({0: "NO", 1: ">30", 2: "<30"}))

# ============================================================
# 3. FEATURE SCALING & TRAIN/TEST SPLIT
# ============================================================
y = data["readmitted"]
X = data.drop(columns="readmitted")

feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set: {X_train_scaled.shape[0]} samples")
print(f"Test set:  {X_test_scaled.shape[0]} samples")

# ============================================================
# 4. MODEL COMPARISON — 5-fold cross-validation
# ============================================================
print(f"\n{'='*60}")
print(f"{c.BOLD}MODEL COMPARISON (5-fold cross-validation){c.END}")
print(f"{'='*60}\n")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
    "Linear SVM":          LinearSVC(max_iter=2000, class_weight="balanced", random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42),
}

scoring = ["accuracy", "f1_weighted"]
cv_results = {}

for name, model in models.items():
    print(f"Training {name} ...", end=" ", flush=True)
    scores = cross_validate(
        model, X_train_scaled, y_train,
        cv=5, scoring=scoring, n_jobs=-1
    )
    acc = scores["test_accuracy"].mean()
    f1 = scores["test_f1_weighted"].mean()
    cv_results[name] = {"accuracy": acc, "f1_weighted": f1}
    print(f"{c.GREEN}done{c.END}  |  Accuracy: {acc:.4f}  |  Weighted F1: {f1:.4f}")

best_model_name = max(cv_results, key=lambda k: cv_results[k]["f1_weighted"])
print(f"\n{c.BOLD}Best model by weighted F1:{c.END} {c.CYAN}{best_model_name}{c.END}")

# ============================================================
# 4b. GRAPHS — per-model results
# ============================================================
target_names = ["NO (0)", ">30 (1)", "<30 (2)"]

# Train each model on full train set, evaluate on test set
test_predictions = {}
test_metrics = {}
for name, model in models.items():
    print(f"Fitting {name} on full train set ...", end=" ", flush=True)
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    test_predictions[name] = preds
    test_metrics[name] = {
        "accuracy": accuracy_score(y_test, preds),
        "f1_weighted": f1_score(y_test, preds, average="weighted"),
        "f1_per_class": f1_score(y_test, preds, average=None),
    }
    print(f"{c.GREEN}done{c.END}")

# --- Graph 1: CV Accuracy & Weighted F1 comparison ---
model_names = list(cv_results.keys())
cv_acc = [cv_results[n]["accuracy"] for n in model_names]
cv_f1 = [cv_results[n]["f1_weighted"] for n in model_names]

x = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - width / 2, cv_acc, width, label="Accuracy")
bars2 = ax.bar(x + width / 2, cv_f1, width, label="Weighted F1")
ax.set_ylabel("Score")
ax.set_title("5-Fold Cross-Validation — Accuracy vs Weighted F1")
ax.set_xticks(x)
ax.set_xticklabels(model_names)
ax.set_ylim(0.4, 0.65)
ax.legend()
for bar_group in [bars1, bars2]:
    for bar in bar_group:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("cv_comparison.png", dpi=150)
plt.show()
print("Saved cv_comparison.png")

# --- Graph 2: Confusion matrix for each model (2x2 grid) ---
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, name in zip(axes.flat, model_names):
    cm = confusion_matrix(y_test, test_predictions[name])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=target_names, yticklabels=target_names)
    ax.set_title(name)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
plt.suptitle("Confusion Matrices — All Models (Test Set)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("confusion_matrices_all.png", dpi=150)
plt.show()
print("Saved confusion_matrices_all.png")

# --- Graph 3: Per-class F1 scores for each model ---
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(target_names))
n_models = len(model_names)
bar_width = 0.8 / n_models
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

for i, name in enumerate(model_names):
    f1_vals = test_metrics[name]["f1_per_class"]
    offset = (i - n_models / 2 + 0.5) * bar_width
    bars = ax.bar(x + offset, f1_vals, bar_width, label=name, color=colors[i])
    for bar in bars:
        height = bar.get_height()
        if height > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, height + 0.005,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=8)

ax.set_ylabel("F1 Score")
ax.set_title("Per-Class F1 Score — All Models (Test Set)")
ax.set_xticks(x)
ax.set_xticklabels(target_names)
ax.set_ylim(0, 0.85)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("per_class_f1.png", dpi=150)
plt.show()
print("Saved per_class_f1.png")

# ============================================================
# 5. PRODUCTION MODEL — Random Forest on full train, eval on test
# ============================================================
print(f"\n{'='*60}")
print(f"{c.BOLD}PRODUCTION MODEL: Random Forest{c.END}")
print(f"{'='*60}\n")

rf = models["Random Forest"]
y_pred = test_predictions["Random Forest"]

acc = test_metrics["Random Forest"]["accuracy"]
f1 = test_metrics["Random Forest"]["f1_weighted"]
print(f"Test Accuracy:    {acc:.4f}")
print(f"Test Weighted F1: {f1:.4f}\n")

print(f"{c.BOLD}Classification Report:{c.END}")
print(classification_report(y_test, y_pred, target_names=target_names))

# Production confusion matrix (standalone)
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=target_names, yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Random Forest (Production)")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150)
plt.show()
print("Saved confusion_matrix.png")

# Top-10 Feature Importances
importances = rf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

print(f"\n{c.BOLD}Top 10 Feature Importances:{c.END}")
for rank, idx in enumerate(indices, 1):
    print(f"  {rank:2d}. {feature_names[idx]:30s}  {importances[idx]:.4f}")

plt.figure(figsize=(10, 5))
top_features = [feature_names[i] for i in indices]
top_importances = importances[indices]
plt.barh(range(10), top_importances[::-1])
plt.yticks(range(10), top_features[::-1])
plt.xlabel("Importance")
plt.title("Top 10 Feature Importances — Random Forest")
plt.tight_layout()
plt.savefig("feature_importances.png", dpi=150)
plt.show()
print("Saved feature_importances.png")

