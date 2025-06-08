import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import xgboost

print("XGBoost version:", xgboost.__version__)

# === Load and preprocess data ===
df = pd.read_csv("heatpump_augmented_golf.csv")
df.replace(['Unknown', 'unknown'], np.nan, inplace=True)
df.drop(columns=[col for col in ['CommissionedAt', 'CommissionedMonth'] if col in df.columns], inplace=True, errors='ignore')
df.dropna(inplace=True)
df.drop(columns='SerialNumber', inplace=True, errors='ignore')

potential_cats = ['BoilerType', 'DhwType', 'Model',
                  'first_operation_type', 'last_operation_type', 'dominant_operation_type']
categorical_cols = [col for col in potential_cats if col in df.columns]
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=False)
df_encoded[df_encoded.select_dtypes(include='object').columns] = df_encoded.select_dtypes(include='object').apply(pd.to_numeric)

X = df_encoded.drop('Target_Broken', axis=1)
y = df_encoded['Target_Broken'].astype(int)

# === Feature selection ===
from sklearn.feature_selection import SelectFromModel
rf_selector = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf_selector.fit(X, y)
selector_rf = SelectFromModel(rf_selector, threshold="mean", prefit=True)
X_selected_rf = pd.DataFrame(selector_rf.transform(X), columns=X.columns[selector_rf.get_support()], index=X.index)

# === Metric functions ===
def compute_rates(cm):
    TN, FP, FN, TP = cm.ravel()
    total = TP + TN + FP + FN
    fn_rate = FN / total
    pos_rate = (TP + FP) / total
    return fn_rate, pos_rate

def compute_full_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    total = TP + TN + FP + FN
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec0 = TN / (TN + FN) if (TN + FN) > 0 else 0
    rec0 = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_0 = 2 * prec0 * rec0 / (prec0 + rec0) if (prec0 + rec0) > 0 else 0
    macro_f1 = (f1 + f1_0) / 2
    return acc, prec, rec, f1, macro_f1, cm

# === Undersampling experiments ===
ratios = [(round(1 - p, 2), round(p, 2)) for p in np.arange(0.05, 0.96, 0.01)]
n_runs = 20
avg_fn_rates = []
avg_pos_rates = []

for maj_ratio, min_ratio in ratios:
    fn_rate_runs = []
    pos_rate_runs = []

    for run in range(n_runs):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
        for train_idx, test_idx in skf.split(X_selected_rf, y):
            X_train, X_test = X_selected_rf.iloc[train_idx], X_selected_rf.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            min_count = sum(y_train == 1)
            target_maj = int(min_count * (maj_ratio / min_ratio))
            rus = RandomUnderSampler(sampling_strategy={0: target_maj, 1: min_count}, random_state=run)
            X_train_us, y_train_us = rus.fit_resample(X_train, y_train)

            model = RandomForestClassifier(n_estimators=200, random_state=run)
            model.fit(X_train_us, y_train_us)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            fn_rate, pos_rate = compute_rates(cm)
            fn_rate_runs.append(fn_rate)
            pos_rate_runs.append(pos_rate)

    avg_fn_rates.append(np.mean(fn_rate_runs))
    avg_pos_rates.append(np.mean(pos_rate_runs))

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(avg_pos_rates, avg_fn_rates, marker='o', label="Undersample Model (20 runs)")
plt.axhline(y=0.01, color='red', linestyle='--', label='FN Rate = 0.01')
plt.xlabel("Positive Rate")
plt.ylabel("False Negative Rate")
plt.title("Avg FN Rate vs Positive Rate across Undersample Ratios with Feature Selection (20 runs x 5 folds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Best ratios table ===
results_df = pd.DataFrame({
    "Majority Ratio": [r[0] for r in ratios],
    "Minority Ratio": [r[1] for r in ratios],
    "Avg FN Rate": avg_fn_rates,
    "Avg Positive Rate": avg_pos_rates
})
filtered_df = results_df[results_df["Avg FN Rate"] <= 0.01].sort_values(by="Avg Positive Rate")
print("Best undersampling ratios with Avg FN Rate <= 0.01 and lowest Avg Positive Rate:")
print(filtered_df.head(10))

# === Evaluate detailed metrics for best ratio ===
best_maj_ratio = filtered_df.iloc[0]["Majority Ratio"]
best_min_ratio = filtered_df.iloc[0]["Minority Ratio"]
print(f"\nEvaluating detailed metrics for best ratio: {best_maj_ratio}:{best_min_ratio}")

accs, precs, recs, f1s, macro_f1s = [], [], [], [], []
conf_matrix_total = np.zeros((2, 2))

for run in range(n_runs):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
    for train_idx, test_idx in skf.split(X_selected_rf, y):
        X_train, X_test = X_selected_rf.iloc[train_idx], X_selected_rf.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        min_count = sum(y_train == 1)
        target_maj = int(min_count * (best_maj_ratio / best_min_ratio))
        rus = RandomUnderSampler(sampling_strategy={0: target_maj, 1: min_count}, random_state=run)
        X_train_us, y_train_us = rus.fit_resample(X_train, y_train)

        model = RandomForestClassifier(n_estimators=100, random_state=run)
        model.fit(X_train_us, y_train_us)
        y_pred = model.predict(X_test)

        acc, prec, rec, f1, macro_f1, cm = compute_full_metrics(y_test, y_pred)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        macro_f1s.append(macro_f1)
        conf_matrix_total += cm

# === Final Results ===
avg_conf_matrix = (conf_matrix_total / (n_runs * 5)).round(2)
conf_percent = (avg_conf_matrix / avg_conf_matrix.sum(axis=1, keepdims=True) * 100).round(2)

print("\nAverage Confusion Matrix (Counts):")
print(avg_conf_matrix)

print("\nAverage Confusion Matrix (Percentages):")
print(conf_percent)

print("\nAverage Performance Metrics:")
print(f"Accuracy:  {np.mean(accs):.3f}")
print(f"Precision: {np.mean(precs):.3f}")
print(f"Recall:    {np.mean(recs):.3f}")
print(f"F1 Score:  {np.mean(f1s):.3f}")
print(f"Macro F1:  {np.mean(macro_f1s):.3f}")
