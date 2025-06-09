import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Load data
df = pd.read_csv("heatpump_svm_ready.csv")
X = df.drop(columns=["Target_Broken"])
y = df["Target_Broken"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Identify column types
numeric_cols = [col for col in X.columns if "_" not in col]
one_hot_cols = [col for col in X.columns if "_" in col]

# Define imputers
numeric_imputer = SimpleImputer(strategy="median")
onehot_imputer = SimpleImputer(strategy="constant", fill_value=0)

# Column transformer for imputation
preprocessor = ColumnTransformer([
    ("num", numeric_imputer, numeric_cols),
    ("cat", onehot_imputer, one_hot_cols),
])

# Pipeline
pipeline = ImbPipeline(steps=[
    ("impute", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("feature_select", SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear"))),
    ("svm", SVC(probability=True, class_weight="balanced", random_state=42))
])

# Hyperparameter grid
param_grid = {
    "svm__C": [0.01, 0.1, 1, 10],
    "svm__kernel": ["linear", "rbf"]
}

# Grid search
grid = GridSearchCV(pipeline, param_grid, scoring="roc_auc", cv=5, n_jobs=-1)
grid.fit(X_train, y_train)

# Evaluation
y_prob = grid.predict_proba(X_test)[:, 1]
y_pred = grid.predict(X_test)

print("Best parameters:", grid.best_params_)
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label="SVM ROC")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM with SMOTE + Feature Selection")
plt.grid(True)
plt.legend()
plt.show()
