import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np

# Load preprocessed data
df = pd.read_csv("heatpump_svm_ready.csv")
X = df.drop(columns=["Target_Broken"])
y = df["Target_Broken"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Identify columns
numeric_cols = [col for col in X.columns if "_" not in col]
one_hot_cols = [col for col in X.columns if "_" in col]

# Imputation strategies
numeric_imputer = SimpleImputer(strategy="median")
onehot_imputer = SimpleImputer(strategy="constant", fill_value=0)
preprocessor = ColumnTransformer([
    ("num", numeric_imputer, numeric_cols),
    ("cat", onehot_imputer, one_hot_cols),
])

# Pipeline with reduced SMOTE
pipeline = ImbPipeline(steps=[
    ("impute", preprocessor),
    ("feature_select", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear"))),
    ("smote", SMOTE(sampling_strategy=0.5, random_state=42)),
    ("svm", SVC(kernel="linear", probability=True, class_weight="balanced", C=0.1, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Predict probabilities
y_prob = pipeline.predict_proba(X_test)[:, 1]

# Threshold optimization
precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
best_thresh = thresholds[np.argmax(f1_scores)]
print("Best threshold for F1:", best_thresh)

# Predict with optimized threshold
y_pred_thresh = (y_prob >= best_thresh).astype(int)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred_thresh, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_thresh))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Plot precision-recall curve
plt.plot(recall, precision, label="Precision-Recall curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.grid(True)
plt.legend()
plt.show()
