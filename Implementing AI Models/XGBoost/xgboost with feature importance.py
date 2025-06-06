import pandas as pd
import numpy as np
from xgboost import XGBClassifier, plot_importance
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# === Load dataset ===
df = pd.read_csv("heatpump_augmented_mika.csv")

# === Prepare features and target ===
X = df.drop(columns=["Target_Broken", "State"])
y = df["Target_Broken"]

# Convert object columns to category codes
for col in X.select_dtypes(include=['object']).columns:
    X[col] = X[col].astype("category").cat.codes

# Fill missing values
X = X.fillna(-999)

# === Split ===
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.15, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1765, stratify=y_train_val, random_state=42)

# === Apply SMOTE ===
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print("Before SMOTE:", np.bincount(y_train))
print("After SMOTE:", np.bincount(y_train_smote))

# === Train XGBoost model ===
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    eval_metric="logloss",
    scale_pos_weight=1.0,
    random_state=42
)
model.fit(X_train_smote, y_train_smote)

# === Predict ===
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Optional: adjust threshold
threshold = 0.2
y_pred = (y_prob >= threshold).astype(int)

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

# === Feature importance ===
plt.figure(figsize=(10, 8))
plot_importance(model, max_num_features=20, importance_type='gain', title="Top 20 Important Features")
plt.tight_layout()
plt.show()
