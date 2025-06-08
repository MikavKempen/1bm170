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
metrics_weighted = []

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

    # 3.d. Class-weighted Random Forest
    model_weighted = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model_weighted.fit(X_train, y_train)

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
    y_pred_weighted = model_weighted.predict(X_test)

    # Compute confusion matrices
    cm_sm = confusion_matrix(y_test, y_pred_sm)
    cm_us = confusion_matrix(y_test, y_pred_us)
    cm_orig = confusion_matrix(y_test, y_pred_orig)
    cm_weighted = confusion_matrix(y_test, y_pred_weighted)

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
    plot_confusion_matrix_with_percentages(cm_weighted, "Random Forest (Class Weighted)")

    # Extract metrics
    def compute_metrics(cm):
        TN, FP, FN, TP = cm.ravel()
        total = TP + TN + FP + FN

        acc = (TP + TN) / total
        prec = TP / (TP + FP) if (TP + FP) > 0 else 0
        rec = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

        prec0 = TN / (TN + FN) if (TN + FN) > 0 else 0
        rec0 = TN / (TN + FP) if (TN + FP) > 0 else 0
        f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0
        macro_f1 = (f1 + f1_0) / 2

        fn_rate = FN / total
        pos_rate = (TP + FP) / total

        return acc, prec, rec, f1, macro_f1, fn_rate, pos_rate

    # Store and print evaluation metrics
    for name, cm, store in zip(["SMOTE", "Undersample", "Original", "Weighted"], [cm_sm, cm_us, cm_orig, cm_weighted], [metrics_smote, metrics_under, metrics_orig, metrics_weighted]):
        acc, prec, rec, f1s, macro, fn_rate, pos_rate = compute_metrics(cm)
        store.append([acc, prec, rec, f1s, macro, fn_rate, pos_rate])
        print(f"{name} model - Accuracy: {acc:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1s:.3f}, Macro F1: {macro:.3f}")

    fold += 1

# Compute average metrics across folds
for name, results in zip(["SMOTE", "Undersample", "Original", "Weighted"], [metrics_smote, metrics_under, metrics_orig, metrics_weighted]):
    avg_metrics = np.mean(results, axis=0)
    print(f"\n{name} model average over 5 folds:")
    print(f"Accuracy: {avg_metrics[0]:.3f}, Precision: {avg_metrics[1]:.3f}, Recall: {avg_metrics[2]:.3f}, F1: {avg_metrics[3]:.3f}, Macro F1: {avg_metrics[4]:.3f}, FN Rate: {avg_metrics[5]:.4f}, Pos Rate: {avg_metrics[6]:.4f}")

# SHAP analysis for class-weighted model
final_model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
final_model.fit(X, y)

explainer = shap.TreeExplainer(final_model)
X_sample = X.sample(n=500, random_state=42) if X.shape[0] > 500 else X
shap_values = explainer.shap_values(X_sample)

if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_sample, plot_type="bar")
    shap.summary_plot(shap_values[1], X_sample)
else:
    shap.summary_plot(shap_values, X_sample, plot_type="bar")
    shap.summary_plot(shap_values, X_sample)

importances = final_model.feature_importances_
feature_names = X.columns
feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}).sort_values(by='Importance', ascending=False).head(20)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_imp_df, color='skyblue')
plt.title("Top 20 Feature Importances from Class-Weighted Random Forest")
plt.tight_layout()
plt.show()
