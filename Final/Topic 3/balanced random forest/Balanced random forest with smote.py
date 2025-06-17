import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# === Load dataset ===
df = pd.read_csv("heatpump_augmented_mika.csv")

# === Drop target leaks and irrelevant columns ===
df = df.drop(columns=["State", "SerialNumber", "ID"], errors="ignore")

# === Select top 15 features from previous ranking ===
top_features = [
    'CommissionedAt',
    'CommissionedMonth',
    'NumberOfFailedTests',
    'VoltageWithstandAc',
    'GroundContinuityEvaporator',
    'BoilerType',
    'total_process_timespan_seconds',
    'Vacuum not reached to soon',
    'unique_outlier_types',
    'total_duration',
    'GroundContinuityFan',
    'DhwType',
    'has_impossible_timing',
    'total_process_duration',
    'OpPlannedTotalQuantity'
]

X = df[top_features].copy()
y = df["Target_Broken"]

# === Encode object columns ===
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype("category").cat.codes

# === Handle missing values ===
X = X.fillna(-999)

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, stratify=y, random_state=42
)

# === Apply SMOTE ===
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# === Train Random Forest ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_smote, y_train_smote)

# === Predictions ===
y_pred = clf.predict(X_test)
y_prob = clf.predict_proba(X_test)[:, 1]

# === Metrics ===
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# === Confusion Matrix ===
cm = confusion_matrix(y_test, y_pred)
labels = np.array([['True Negative', 'False Positive'], ['False Negative', 'True Positive']])
counts = cm.astype(str)
annot = np.char.add(labels, '\n' + counts)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()
