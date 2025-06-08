import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# === Load and preprocess data ===
df = pd.read_csv("heatpump_augmented_golf.csv")
df.replace(['Unknown', 'unknown'], np.nan, inplace=True)
df.drop(columns=[col for col in ['CommissionedAt', 'CommissionedMonth'] if col in df.columns], inplace=True, errors='ignore')
df.dropna(inplace=True)
df.drop(columns='SerialNumber', inplace=True, errors='ignore')

potential_cats = ['BoilerType', 'DhwType', 'Model', 'first_operation_type', 'last_operation_type', 'dominant_operation_type']
categorical_cols = [col for col in potential_cats if col in df.columns]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
df_encoded[df_encoded.select_dtypes(include='object').columns] = df_encoded.select_dtypes(include='object').apply(pd.to_numeric)

X = df_encoded.drop('Target_Broken', axis=1)
y = df_encoded['Target_Broken'].astype(int)

# === Utility functions ===
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

# === Undersampling experiments ===
ratios = [(0.55, 0.45), (0.56, 0.44), (0.57, 0.43), (0.58, 0.42), (0.59, 0.41), (0.6, 0.4), (0.61, 0.39), (0.62, 0.38), (0.63, 0.37), (0.64, 0.36), (0.65, 0.35), (0.66, 0.34), (0.67, 0.33), (0.68, 0.32), (0.69, 0.31)]
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for maj_ratio, min_ratio in ratios:
    print(f"\n=== Undersampling at ratio {int(maj_ratio*100)}:{int(min_ratio*100)} ===")
    undersample_metrics = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        min_count = sum(y_train == 1)
        target_maj = int(min_count * (maj_ratio / min_ratio))
        rus = RandomUnderSampler(sampling_strategy={0: target_maj, 1: min_count}, random_state=42)
        X_train_us, y_train_us = rus.fit_resample(X_train, y_train)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train_us, y_train_us)
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        undersample_metrics.append(compute_metrics(cm))

        plot_confusion_matrix_with_percentages(cm, f"Undersample - Fold {fold} - Ratio {int(maj_ratio*100)}:{int(min_ratio*100)}")
        plot_feature_importance(model, X.columns, f"Undersample Features - Fold {fold} - Ratio {int(maj_ratio*100)}:{int(min_ratio*100)}")

    avg = np.mean(undersample_metrics, axis=0)
    print(f"\n>>> Averages for ratio {int(maj_ratio*100)}:{int(min_ratio*100)}")
    print(f"Undersample - Acc: {avg[0]:.3f}, Prec: {avg[1]:.3f}, Rec: {avg[2]:.3f}, F1: {avg[3]:.3f}, Macro F1: {avg[4]:.3f}, FN Rate: {avg[5]:.3f}, Pos Rate: {avg[6]:.3f}")

