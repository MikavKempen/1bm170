import pandas as pd
import numpy as np

import xgboost
print("XGBoost version:", xgboost.__version__)

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
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def plot_confusion_matrix_with_percentages(cm, title):
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = np.array([[f"{cm[i,j]}\n{cm_percent[i,j]:.2f}%" for j in range(2)] for i in range(2)])
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm_percent, annot=labels, fmt="", cmap="Blues", cbar=False,
                xticklabels=["0", "1"], yticklabels=["0", "1"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(title)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, title):
    importances = model.feature_importances_
    feat_imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances}) \
                    .sort_values(by='Importance', ascending=False).head(20)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, color='skyblue')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def compute_metrics(cm):
    TN, FP, FN, TP = cm.ravel()
    acc = (TP + TN) / (TP + TN + FP + FN)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0
    rec = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    prec0 = TN / (TN + FN) if (TN + FN) > 0 else 0
    rec0 = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0
    macro_f1 = (f1 + f1_0) / 2
    return acc, prec, rec, f1, macro_f1

# Define class ratios to test
ratios = [(0.725, 0.275), (0.75, 0.25), (0.775, 0.225), (0.8, 0.2), (0.825, 0.175), (0.85, 0.15), (0.875, 0.125), (0.9, 0.1)]

# 5-fold stratified CV
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Loop over each ratio
for maj_ratio, min_ratio in ratios:
    print(f"\n=== SMOTE & Undersample at ratio {int(maj_ratio*100)}:{int(min_ratio*100)} ===")

    smote_metrics = []
    undersample_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        maj_count = sum(y_train == 0)
        min_count = sum(y_train == 1)

        # SMOTE
        smote_ratio = min_ratio / maj_ratio
        sm = SMOTE(sampling_strategy=smote_ratio, random_state=42)
        X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
        model_sm = RandomForestClassifier(n_estimators=100, random_state=42)
        model_sm.fit(X_train_sm, y_train_sm)
        y_pred_sm = model_sm.predict(X_test)
        cm_sm = confusion_matrix(y_test, y_pred_sm)
        smote_metrics.append(compute_metrics(cm_sm))
        plot_confusion_matrix_with_percentages(cm_sm, f"SMOTE - Fold {fold} - Ratio {int(maj_ratio*100)}:{int(min_ratio*100)}")
        plot_feature_importance(model_sm, X.columns, f"SMOTE Features - Fold {fold} - Ratio {int(maj_ratio*100)}:{int(min_ratio*100)}")

        # Undersampling
        target_min = min_count
        target_maj = int(target_min * (maj_ratio / min_ratio))
        rus = RandomUnderSampler(sampling_strategy={0: target_maj, 1: target_min}, random_state=42)
        X_train_us, y_train_us = rus.fit_resample(X_train, y_train)
        model_us = RandomForestClassifier(n_estimators=100, random_state=42)
        model_us.fit(X_train_us, y_train_us)
        y_pred_us = model_us.predict(X_test)
        cm_us = confusion_matrix(y_test, y_pred_us)
        undersample_metrics.append(compute_metrics(cm_us))
        plot_confusion_matrix_with_percentages(cm_us, f"Undersample - Fold {fold} - Ratio {int(maj_ratio*100)}:{int(min_ratio*100)}")
        plot_feature_importance(model_us, X.columns, f"Undersample Features - Fold {fold} - Ratio {int(maj_ratio*100)}:{int(min_ratio*100)}")

    # Print average metrics
    sm_avg = np.mean(smote_metrics, axis=0)
    us_avg = np.mean(undersample_metrics, axis=0)
    print(f"\n>>> Averages for ratio {int(maj_ratio*100)}:{int(min_ratio*100)}")
    print(f"SMOTE       - Acc: {sm_avg[0]:.3f}, Prec: {sm_avg[1]:.3f}, Rec: {sm_avg[2]:.3f}, F1: {sm_avg[3]:.3f}, Macro F1: {sm_avg[4]:.3f}")
    print(f"Undersample - Acc: {us_avg[0]:.3f}, Prec: {us_avg[1]:.3f}, Rec: {us_avg[2]:.3f}, F1: {us_avg[3]:.3f}, Macro F1: {us_avg[4]:.3f}")

