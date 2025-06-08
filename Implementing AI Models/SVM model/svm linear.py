import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.combine import SMOTETomek
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("heatpump_svm_ready.csv")
X = df.drop(columns=["Target_Broken"])
y = df["Target_Broken"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Identify numeric vs one-hot columns
numeric_cols = [col for col in X.columns if '_' not in col]
one_hot_cols = [col for col in X.columns if '_' in col]

# Imputers
numeric_imputer = SimpleImputer(strategy="median")
onehot_imputer = SimpleImputer(strategy="constant", fill_value=0)

preprocessor = ColumnTransformer([
    ("num", numeric_imputer, numeric_cols),
    ("cat", onehot_imputer, one_hot_cols),
])

# Final pipeline with linear SVM and SMOTETomek
pipeline = ImbPipeline(steps=[
    ("impute", preprocessor),
    ("resample", SMOTETomek(random_state=42)),
    ("svm", SVC(kernel="linear", probability=True, class_weight="balanced", random_state=42)),
])

# Fit
pipeline.fit(X_train, y_train)

# Predict
y_prob = pipeline.predict_proba(X_test)[:, 1]
y_pred = pipeline.predict(X_test)

# Evaluate
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# Plot ROC
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label="SVM ROC")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - SVM")
plt.grid(True)
plt.legend()
plt.show()
