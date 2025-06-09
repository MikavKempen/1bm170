import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

# === Load dataset ===
df = pd.read_csv("heatpump_augmented_mika.csv")

# === Prepare features and target ===
X = df.drop(columns=["Target_Broken", "State"], errors="ignore")
y = df["Target_Broken"]

# Encode object columns
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype("category").cat.codes

X = X.fillna(-999)

# === Split into train/test sets ===
X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)

# === Custom resampling ===
# 67% of class 0, 33% of class 1
X_train_full["Target_Broken"] = y_train_full.values
broken = X_train_full[X_train_full["Target_Broken"] == 1]
not_broken = X_train_full[X_train_full["Target_Broken"] == 0]

# Sample class 0 (undersample to ~2x the class 1 size)
not_broken_sampled = not_broken.sample(frac=0.66, random_state=42)
# Sample class 1 (oversample to balance)
broken_oversampled = broken.sample(n=len(not_broken_sampled)//2, replace=True, random_state=42)

# Combine and shuffle
resampled = pd.concat([not_broken_sampled, broken_oversampled]).sample(frac=1, random_state=42)
y_train = resampled["Target_Broken"]
X_train = resampled.drop(columns=["Target_Broken"])

# === Train XGBoost model ===
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42
)
model.fit(X_train, y_train)

# === Predictions and probabilities ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# === Metrics ===
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, zero_division=0))
print("Recall:", recall_score(y_test, y_pred, zero_division=0))
print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# === Confusion matrix ===
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

# === ROC curve ===
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f'XGBoost (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()
