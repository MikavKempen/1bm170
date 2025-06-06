import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("heatpump_augmented_golf.csv")

# Drop CommissionedAt and CommissionedMonth if they contain any unknown values
for col in ['CommissionedAt', 'CommissionedMonth']:
    if col in df.columns:
        # Check if 'Unknown' appears in the column
        if ((df[col] == 'Unknown') | (df[col] == 'unknown')).any():
            df.drop(columns=col, inplace=True)

# Replace 'Unknown' string placeholders with actual NaN for uniform handling
df.replace(['Unknown', 'unknown'], np.nan, inplace=True)

# Drop any rows that have any NaN (after the replacements above)
df.dropna(inplace=True)

print("Data shape after cleaning:", df.shape)

# Drop the SerialNumber column (unique identifier, not useful for modeling)
if 'SerialNumber' in df.columns:
    df.drop(columns='SerialNumber', inplace=True)

# Identify categorical columns that need encoding
categorical_cols = []
# Known categorical fields (by domain knowledge or inspection):
potential_cats = ['BoilerType', 'DhwType', 'Model',
                  'first_operation_type', 'last_operation_type', 'dominant_operation_type']
for col in potential_cats:
    if col in df.columns:
        categorical_cols.append(col)

# One-hot encode categorical features
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

# Convert any remaining numeric columns that are of object type to floats
for col in df_encoded.select_dtypes(include='object').columns:
    df_encoded[col] = pd.to_numeric(df_encoded[col])

# Separate features and target
X = df_encoded.drop('Target_Broken', axis=1)
y = df_encoded['Target_Broken'].astype(int)

print("Features shape after encoding:", X.shape)
print("Example columns:", X.columns[:5].tolist())

# Examine class imbalance
print("Class distribution in full data:", np.bincount(y))

from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import shap

# Initialize stratified 5-fold CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Prepare to collect results
fold = 1
metrics_smote = []
metrics_under = []
metrics_orig = []

for train_idx, test_idx in skf.split(X, y):
    # Split data into train and test for this fold
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print(f"\n*** Fold {fold} ***")
    print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)} (Broken in test: {sum(y_test)})")

    # 3.a. SMOTE Oversampling on training set
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    print("After SMOTE oversampling - training class counts:", np.bincount(y_train_sm))

    # 3.b. Random Undersampling on training set
    rus = RandomUnderSampler(random_state=42)
    X_train_us, y_train_us = rus.fit_resample(X_train, y_train)
    print("After undersampling - training class counts:", np.bincount(y_train_us))

    # 3.c. No resampling (original training data)
    print("Original training class counts:", np.bincount(y_train))

    # 4. Train Random Forest on each training set
    model_smote = RandomForestClassifier(n_estimators=100, random_state=42)
    model_smote.fit(X_train_sm, y_train_sm)

    model_under = RandomForestClassifier(n_estimators=100, random_state=42)
    model_under.fit(X_train_us, y_train_us)

    model_orig = RandomForestClassifier(n_estimators=100, random_state=42)
    model_orig.fit(X_train, y_train)

    # 5. Make predictions on the test set for each model
    y_pred_sm = model_smote.predict(X_test)
    y_pred_us = model_under.predict(X_test)
    y_pred_orig = model_orig.predict(X_test)

    # Compute confusion matrices
    cm_sm = confusion_matrix(y_test, y_pred_sm)
    cm_us = confusion_matrix(y_test, y_pred_us)
    cm_orig = confusion_matrix(y_test, y_pred_orig)

    def plot_confusion_matrix_with_percentages(cm, title):
        total = cm.sum()
        cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
        labels = np.array([[f"{cm[i,j]}\n{cm_percent[i,j]:.2f}%" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
        plt.figure(figsize=(4, 4))
        sns.heatmap(cm_percent, annot=labels, fmt="", cmap="Blues", cbar=False, xticklabels=["0", "1"], yticklabels=["0", "1"])
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(title)
        plt.tight_layout()
        plt.show()

    plot_confusion_matrix_with_percentages(cm_sm, "Random Forest (SMOTE)")
    plot_confusion_matrix_with_percentages(cm_us, "Random Forest (Undersample)")
    plot_confusion_matrix_with_percentages(cm_orig, "Random Forest (Original)")

    # Extract metrics
    def compute_metrics(cm):
        TN, FP, FN, TP = cm.ravel()
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precision0 = TN / (TN + FN) if (TN + FN) > 0 else 0.0
        recall0 = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        f1_0 = 2 * precision0 * recall0 / (precision0 + recall0) if (precision0 + recall0) > 0 else 0.0
        macro_f1 = (f1 + f1_0) / 2
        return accuracy, precision, recall, f1, macro_f1

    # Store and print evaluation metrics
    for name, cm, store in zip(["SMOTE", "Undersample", "Original"], [cm_sm, cm_us, cm_orig], [metrics_smote, metrics_under, metrics_orig]):
        acc, prec, rec, f1s, macro = compute_metrics(cm)
        store.append([acc, prec, rec, f1s, macro])
        print(f"{name} model - Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1s:.3f}, Macro F1: {macro:.3f}")

    # 6. ROC Curve and AUC for each model
    y_prob_sm = model_smote.predict_proba(X_test)[:,1]
    y_prob_us = model_under.predict_proba(X_test)[:,1]
    y_prob_orig = model_orig.predict_proba(X_test)[:,1]
    fpr_sm, tpr_sm, _ = roc_curve(y_test, y_prob_sm)
    fpr_us, tpr_us, _ = roc_curve(y_test, y_prob_us)
    fpr_orig, tpr_orig, _ = roc_curve(y_test, y_prob_orig)
    auc_sm = roc_auc_score(y_test, y_prob_sm)
    auc_us = roc_auc_score(y_test, y_prob_us)
    auc_orig = roc_auc_score(y_test, y_prob_orig)
    print(f"ROC AUC (SMOTE model): {auc_sm:.3f}, ROC AUC (Undersample model): {auc_us:.3f}, ROC AUC (Original model): {auc_orig:.3f}")

    # Plot ROC curve for this fold
    plt.figure()
    plt.plot(fpr_sm, tpr_sm, label=f"SMOTE (AUC={auc_sm:.2f})")
    plt.plot(fpr_us, tpr_us, label=f"Undersample (AUC={auc_us:.2f})", linestyle='--')
    plt.plot(fpr_orig, tpr_orig, label=f"Original (AUC={auc_orig:.2f})", linestyle=':')
    plt.plot([0,1], [0,1], 'k--', alpha=0.7)  # diagonal line
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - Fold {fold}")
    plt.legend(loc="lower right")
    plt.show()

    fold += 1

# Compute average metrics across folds
for name, results in zip(["SMOTE", "Undersample", "Original"], [metrics_smote, metrics_under, metrics_orig]):
    avg_metrics = np.mean(results, axis=0)
    print(f"\n{name} model average over 5 folds:")
    print(f"Accuracy: {avg_metrics[0]:.3f}, Precision: {avg_metrics[1]:.3f}, Recall: {avg_metrics[2]:.3f}, F1: {avg_metrics[3]:.3f}, Macro F1: {avg_metrics[4]:.3f}")

# ===== Feature importance with SHAP on final model (Original training data) =====
# Retrain model_orig on entire dataset
final_model = RandomForestClassifier(n_estimators=100, random_state=42)
final_model.fit(X, y)

# Create SHAP explainer
explainer = shap.TreeExplainer(final_model)

# Sample 500 instances for more stable SHAP plot
X_sample = X.sample(n=500, random_state=42) if X.shape[0] > 500 else X
shap_values = explainer.shap_values(X_sample)

# Robust SHAP plot handling for binary classification
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_sample, plot_type="bar")
    shap.summary_plot(shap_values[1], X_sample)
else:
    shap.summary_plot(shap_values, X_sample, plot_type="bar")
    shap.summary_plot(shap_values, X_sample)

# ===== Classic Random Forest Feature Importances Bar Plot =====
importances = final_model.feature_importances_
feature_names = X.columns

feat_imp_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, color='skyblue')
plt.title("Top 20 Feature Importances from Best Random Forest Classifier")
plt.tight_layout()
plt.show()