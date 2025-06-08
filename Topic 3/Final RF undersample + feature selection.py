import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt

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

X_full = df_encoded.drop('Target_Broken', axis=1)
y = df_encoded['Target_Broken'].astype(int)

# === Metric computation ===
def compute_full_metrics(cm):
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
    return acc, prec, rec, f1, macro_f1, fn_rate, pos_rate, np.array([[TN, FP], [FN, TP]])

# === Phase 1: Run all ratios ===
ratios = [(round(1 - p, 2), round(p, 2)) for p in np.arange(0.05, 0.96, 0.01)]
n_runs = 20

avg_fn_rates = []
avg_pos_rates = []
all_metrics = []
all_feature_importances = {}

for idx, (maj_ratio, min_ratio) in enumerate(ratios):
    fn_rate_runs, pos_rate_runs = [], []
    confusion_sum = np.zeros((2, 2))
    accs, precs, recs, f1s, macro_f1s = [], [], [], [], []
    total_importance = np.zeros(X_full.shape[1])

    for run in range(n_runs):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
        for train_idx, test_idx in skf.split(X_full, y):
            X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            min_count = sum(y_train == 1)
            target_maj = int(min_count * (maj_ratio / min_ratio))
            rus = RandomUnderSampler(sampling_strategy={0: target_maj, 1: min_count}, random_state=run)
            X_train_us, y_train_us = rus.fit_resample(X_train, y_train)

            model = RandomForestClassifier(n_estimators=200, random_state=run)
            model.fit(X_train_us, y_train_us)
            y_pred = model.predict(X_test)

            cm = confusion_matrix(y_test, y_pred)
            acc, prec, rec, f1, macro_f1, fn_rate, pos_rate, cm_vals = compute_full_metrics(cm)
            confusion_sum += cm_vals
            fn_rate_runs.append(fn_rate)
            pos_rate_runs.append(pos_rate)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
            macro_f1s.append(macro_f1)
            total_importance += model.feature_importances_

    avg_fn_rates.append(np.mean(fn_rate_runs))
    avg_pos_rates.append(np.mean(pos_rate_runs))
    all_metrics.append({
        "acc": np.mean(accs),
        "prec": np.mean(precs),
        "rec": np.mean(recs),
        "f1": np.mean(f1s),
        "macro_f1": np.mean(macro_f1s),
        "conf_matrix": (confusion_sum / (n_runs * 5)).round(2)
    })
    all_feature_importances[idx] = total_importance / (n_runs * 5)

# === Phase 1 Plots and Tables ===
plt.figure(figsize=(10, 6))
plt.plot(avg_pos_rates, avg_fn_rates, marker='o', label="Undersample Model (20x5 CV)")
plt.axhline(y=0.01, color='red', linestyle='--', label='FN Rate = 0.01')
plt.xlabel("Positive Rate")
plt.ylabel("False Negative Rate")
plt.title("FN Rate vs Positive Rate (All Features)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

results_df = pd.DataFrame({
    "Majority Ratio": [r[0] for r in ratios],
    "Minority Ratio": [r[1] for r in ratios],
    "Avg FN Rate": avg_fn_rates,
    "Avg Positive Rate": avg_pos_rates
})
filtered_df = results_df[results_df["Avg FN Rate"] <= 0.01].sort_values(by="Avg Positive Rate")
print("Best ratios (FN Rate <= 0.01):")
print(filtered_df.head(10))

# === Phase 2: Use top 30 features from best ratio ===
best_idx = filtered_df.index[0]
top30 = pd.Series(all_feature_importances[best_idx], index=X_full.columns).sort_values(ascending=False).head(30)
X_selected = X_full[top30.index]

# === Re-run with selected features ===
fn_rate_runs, pos_rate_runs = [], []
confusion_sum = np.zeros((2, 2))
accs, precs, recs, f1s, macro_f1s = [], [], [], [], []

for run in range(n_runs):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
    for train_idx, test_idx in skf.split(X_selected, y):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        min_count = sum(y_train == 1)
        target_maj = int(min_count * (ratios[best_idx][0] / ratios[best_idx][1]))
        rus = RandomUnderSampler(sampling_strategy={0: target_maj, 1: min_count}, random_state=run)
        X_train_us, y_train_us = rus.fit_resample(X_train, y_train)

        model = RandomForestClassifier(n_estimators=200, random_state=run)
        model.fit(X_train_us, y_train_us)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        acc, prec, rec, f1, macro_f1, fn_rate, pos_rate, cm_vals = compute_full_metrics(cm)
        confusion_sum += cm_vals
        fn_rate_runs.append(fn_rate)
        pos_rate_runs.append(pos_rate)
        accs.append(acc)
        precs.append(prec)
        recs.append(rec)
        f1s.append(f1)
        macro_f1s.append(macro_f1)

# === Phase 2 Outputs ===
plt.figure(figsize=(10, 6))
plt.plot([np.mean(pos_rate_runs)], [np.mean(fn_rate_runs)], marker='o', color='green', label="Top 30 Feature Model")
plt.axhline(y=0.01, color='red', linestyle='--', label='FN Rate = 0.01')
plt.xlabel("Positive Rate")
plt.ylabel("False Negative Rate")
plt.title("FN Rate vs Positive Rate (Top 30 Features)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

print("\nTop 30 Features and Importances:")
print(top30)

conf_matrix = (confusion_sum / (n_runs * 5)).round(2)
conf_matrix_percent = (conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100).round(2)
print("\nAverage Confusion Matrix (Counts):")
print(conf_matrix)
print("\nAverage Confusion Matrix (Percentages):")
print(conf_matrix_percent)

print("\nAverage Metrics with Top 30 Features:")
print(f"Accuracy: {np.mean(accs):.3f}, Precision: {np.mean(precs):.3f}, Recall: {np.mean(recs):.3f}, "
      f"F1: {np.mean(f1s):.3f}, Macro F1: {np.mean(macro_f1s):.3f}")
