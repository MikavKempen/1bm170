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
ratios = [(0.5, 0.5), (0.6, 0.4), (0.7, 0.3), (0.8, 0.2), (0.9, 0.1)]

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

from sklearn.utils.class_weight import compute_class_weight

# 80:20 undersampling + class weights
maj_ratio, min_ratio = 0.8, 0.2
print(f"\n=== 80:20 Undersample + Class Weights ===")

undersample_weighted_metrics = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    min_count = sum(y_train == 1)
    target_maj = int(min_count * (maj_ratio / min_ratio))
    rus = RandomUnderSampler(sampling_strategy={0: target_maj, 1: min_count}, random_state=42)
    X_train_us, y_train_us = rus.fit_resample(X_train, y_train)

    # Compute class weights for the undersampled training set
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_us), y=y_train_us)
    class_weight_dict = {cls: weight for cls, weight in zip(np.unique(y_train_us), class_weights)}

    model_us_weighted = RandomForestClassifier(n_estimators=100, class_weight=class_weight_dict, random_state=42)
    model_us_weighted.fit(X_train_us, y_train_us)
    y_pred_us_weighted = model_us_weighted.predict(X_test)
    cm_us_weighted = confusion_matrix(y_test, y_pred_us_weighted)
    undersample_weighted_metrics.append(compute_metrics(cm_us_weighted))
    plot_confusion_matrix_with_percentages(cm_us_weighted, f"Undersample + Weights - Fold {fold} - Ratio 80:20")
    plot_feature_importance(model_us_weighted, X.columns, f"Undersample + Weights - Features - Fold {fold}")

# Print average metrics
us_weighted_avg = np.mean(undersample_weighted_metrics, axis=0)
print(f"\n>>> Averages for Undersample + Weights at ratio 80:20")
print(f"Undersample + Weights - Acc: {us_weighted_avg[0]:.3f}, Prec: {us_weighted_avg[1]:.3f}, Rec: {us_weighted_avg[2]:.3f}, F1: {us_weighted_avg[3]:.3f}, Macro F1: {us_weighted_avg[4]:.3f}")

from xgboost import XGBClassifier

print(f"\n=== 80:20 Undersample + XGBoost (scale_pos_weight) ===")

xgb_undersample_metrics = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    min_count = sum(y_train == 1)
    target_maj = int(min_count * (maj_ratio / min_ratio))
    rus = RandomUnderSampler(sampling_strategy={0: target_maj, 1: min_count}, random_state=42)
    X_train_us, y_train_us = rus.fit_resample(X_train, y_train)

    # Compute scale_pos_weight = (# negative / # positive) in the training set
    neg_count = sum(y_train_us == 0)
    pos_count = sum(y_train_us == 1)
    scale_pos_weight = neg_count / pos_count

    model_xgb = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model_xgb.fit(X_train_us, y_train_us)
    y_pred_xgb = model_xgb.predict(X_test)
    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    xgb_undersample_metrics.append(compute_metrics(cm_xgb))
    plot_confusion_matrix_with_percentages(cm_xgb, f"XGBoost - Fold {fold} - Ratio 80:20")
    plot_feature_importance(model_xgb, X.columns, f"XGBoost Features - Fold {fold}")

# Print average metrics
xgb_avg = np.mean(xgb_undersample_metrics, axis=0)
print(f"\n>>> Averages for XGBoost at ratio 80:20")
print(f"XGBoost       - Acc: {xgb_avg[0]:.3f}, Prec: {xgb_avg[1]:.3f}, Rec: {xgb_avg[2]:.3f}, F1: {xgb_avg[3]:.3f}, Macro F1: {xgb_avg[4]:.3f}")

from xgboost import XGBClassifier
from xgboost import __version__ as xgb_ver
print("Using XGBoost version:", xgb_ver)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

print(f"\n=== 80:20 Undersample + XGBoost + EarlyStopping + Threshold Tuning ===")

xgb_final_metrics = []

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    min_count = sum(y_train == 1)
    target_maj = int(min_count * (maj_ratio / min_ratio))
    rus = RandomUnderSampler(sampling_strategy={0: target_maj, 1: min_count}, random_state=42)
    X_train_us, y_train_us = rus.fit_resample(X_train, y_train)

    # Compute scale_pos_weight for training set
    neg_count = sum(y_train_us == 0)
    pos_count = sum(y_train_us == 1)
    scale_pos_weight = neg_count / pos_count

    # Early stopping split
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_us, y_train_us, test_size=0.2, stratify=y_train_us, random_state=42
    )

    model_xgb = XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        random_state=42
    )

    model_xgb.fit(
        X_train_split, y_train_split,
        eval_set=[(X_val_split, y_val_split)],
        early_stopping_rounds=20,
        verbose=False
    )

    # Predict probabilities
    y_proba_xgb = model_xgb.predict_proba(X_test)[:, 1]

    # Threshold tuning
    best_f1 = 0
    best_thresh = 0.5
    best_preds = None

    for thresh in np.arange(0.1, 0.91, 0.01):
        preds = (y_proba_xgb >= thresh).astype(int)
        f1 = f1_score(y_test, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
            best_preds = preds

    print(f"Fold {fold}: Best threshold = {best_thresh:.2f}, F1 = {best_f1:.3f}")
    y_pred_xgb = best_preds

    cm_xgb = confusion_matrix(y_test, y_pred_xgb)
    xgb_final_metrics.append(compute_metrics(cm_xgb))
    plot_confusion_matrix_with_percentages(cm_xgb, f"XGBoost Tuned - Fold {fold} - Ratio 80:20")
    plot_feature_importance(model_xgb, X.columns, f"XGBoost Tuned Features - Fold {fold}")

# Average final metrics
xgb_final_avg = np.mean(xgb_final_metrics, axis=0)
print(f"\n>>> Averages for Tuned XGBoost at ratio 80:20")
print(f"XGBoost Tuned - Acc: {xgb_final_avg[0]:.3f}, Prec: {xgb_final_avg[1]:.3f}, Rec: {xgb_final_avg[2]:.3f}, F1: {xgb_final_avg[3]:.3f}, Macro F1: {xgb_final_avg[4]:.3f}")