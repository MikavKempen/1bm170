import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
from collections import defaultdict

# === Load and preprocess data ===
df = pd.read_csv("heatpump_augmented_final.csv")
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

# === Undersampling experiments ===
ratios = [(round(1 - p, 2), round(p, 2)) for p in np.arange(0.05, 0.96, 0.01)]
n_runs = 20

avg_fn_rates = []
avg_pos_rates = []
all_metrics = []

for maj_ratio, min_ratio in ratios:
    fn_rate_runs, pos_rate_runs = [], []
    confusion_sum = np.zeros((2, 2))
    accs, precs, recs, f1s, macro_f1s = [], [], [], [], []

    for run in range(n_runs):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            min_count = sum(y_train == 1)
            target_maj = int(min_count * (maj_ratio / min_ratio))
            rus = RandomUnderSampler(sampling_strategy={0: target_maj, 1: min_count}, random_state=run)
            X_train_us, y_train_us = rus.fit_resample(X_train, y_train)

            model = RandomForestClassifier(n_estimators=100, random_state=run)
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

# === Plotting ===
plt.figure(figsize=(10, 6))
plt.plot(avg_pos_rates, avg_fn_rates, marker='o', label="Undersample Model (20 runs)")
plt.axhline(y=0.01, color='red', linestyle='--', label='FN Rate = 0.01')
plt.xlabel("Positive Rate")
plt.ylabel("False Negative Rate")
plt.title("Average FN Rate vs Positive Rate across Undersample Ratios (20 runs x 5 folds)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Identify best ratios with FN Rate <= 0.01 ===
results_df = pd.DataFrame({
    "Majority Ratio": [r[0] for r in ratios],
    "Minority Ratio": [r[1] for r in ratios],
    "Avg FN Rate": avg_fn_rates,
    "Avg Positive Rate": avg_pos_rates
})
filtered_df = results_df[results_df["Avg FN Rate"] <= 0.01].sort_values(by="Avg Positive Rate")

# Display the best ratios
print("Best undersampling ratios with Avg FN Rate <= 0.01 and lowest Avg Positive Rate:")
print(filtered_df.head(10))

# === Print final average confusion matrix and performance metrics for best ratio ===
best_idx = filtered_df.index[0]
best_metrics = all_metrics[best_idx]
conf_matrix = best_metrics["conf_matrix"]
conf_matrix_percent = (conf_matrix / conf_matrix.sum(axis=1, keepdims=True) * 100).round(2)

print("Average Confusion Matrix (Counts):")
print(conf_matrix)
print("Average Confusion Matrix (Percentages):")
print(conf_matrix_percent)
print("Average Metrics:")
print(f"Accuracy: {best_metrics['acc']:.3f}, Precision: {best_metrics['prec']:.3f}, Recall: {best_metrics['rec']:.3f}, "
      f"F1: {best_metrics['f1']:.3f}, Macro F1: {best_metrics['macro_f1']:.3f}")

best_idx = filtered_df.index[0]
maj_ratio, min_ratio = ratios[best_idx]
importance_accumulator = defaultdict(float)

for run in range(n_runs):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
    for train_idx, test_idx in skf.split(X, y):
        X_train, _ = X.iloc[train_idx], y.iloc[train_idx]
        y_train = y.iloc[train_idx]

        min_count = sum(y_train == 1)
        target_maj = int(min_count * (maj_ratio / min_ratio))
        rus = RandomUnderSampler(sampling_strategy={0: target_maj, 1: min_count}, random_state=run)
        X_train_us, y_train_us = rus.fit_resample(X_train, y_train)

        model = RandomForestClassifier(n_estimators=200, random_state=run)
        model.fit(X_train_us, y_train_us)

        for feat, importance in zip(X.columns, model.feature_importances_):
            importance_accumulator[feat] += importance

# Normalize by total number of models
total_models = n_runs * 5
avg_importances = {feat: val / total_models for feat, val in importance_accumulator.items()}

# Convert to DataFrame and sort
import_df = pd.DataFrame.from_dict(avg_importances, orient='index', columns=['Importance'])
import_df = import_df.sort_values(by='Importance', ascending=False).head(20)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(import_df.index[::-1], import_df['Importance'][::-1], color='skyblue', edgecolor='black')
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title(f"Undersample Features - Avg over {total_models} models - Ratio {int(maj_ratio*100)}:{int(min_ratio*100)}")
plt.tight_layout()
plt.show()