import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from collections import defaultdict
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"

# === Load and preprocess data ===
df = pd.read_csv("heatpump_nn_ready.csv")
X = df.drop(columns=["Target_Broken"])
y = df["Target_Broken"]

X.fillna(0, inplace=True)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Feature selection (RandomForest-based)
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf.fit(X_scaled, y)
selector = SelectFromModel(rf, prefit=True, threshold='median')
X_selected = selector.transform(X_scaled)

# === Custom loss ===
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return alpha * tf.keras.backend.pow(1. - pt, gamma) * bce
    return loss

# === Metric computation ===
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
    return acc, prec, rec, f1, macro_f1, fn_rate, pos_rate, cm

# === Ratios and experiment config ===
ratios = [(round(1 - p, 2), round(p, 2)) for p in np.arange(0.05, 0.96, 0.01)]
n_runs = 1  # Reduce if training is too slow

avg_fn_rates = []
avg_pos_rates = []
all_metrics = []

for maj_ratio, min_ratio in ratios:
    fn_rates, pos_rates = [], []
    accs, precs, recs, f1s, macro_f1s = [], [], [], [], []
    confusion_sum = np.zeros((2, 2))

    for run in range(n_runs):
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
        for train_idx, test_idx in skf.split(X_selected, y):
            X_train, X_test = X_selected[train_idx], X_selected[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            # Undersample training set
            df_train = pd.DataFrame(X_train)
            df_train['Target_Broken'] = y_train.values
            minor = df_train[df_train['Target_Broken'] == 1]
            major = df_train[df_train['Target_Broken'] == 0]

            min_count = len(minor)
            target_maj = int(min_count * (maj_ratio / min_ratio))
            major_sampled = major.sample(n=target_maj, random_state=run)
            undersampled = pd.concat([minor, major_sampled]).sample(frac=1, random_state=run)

            X_tr = undersampled.drop(columns="Target_Broken").values
            y_tr = undersampled["Target_Broken"].values

            # Class weights
            weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
            class_weights = dict(enumerate(weights))

            # Build model
            model = Sequential([
                Dense(256, input_shape=(X_tr.shape[1],)),
                LeakyReLU(),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128),
                LeakyReLU(),
                BatchNormalization(),
                Dropout(0.4),
                Dense(64),
                LeakyReLU(),
                Dense(1, activation='sigmoid')
            ])
            model.compile(
                optimizer='RMSprop',
                loss=focal_loss(gamma=2, alpha=0.25),
                metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
            )

            callbacks = [
                EarlyStopping(patience=8, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(patience=4, factor=0.5, min_lr=1e-5, verbose=0)
            ]

            model.fit(
                X_tr, y_tr,
                epochs=50,
                batch_size=32,
                validation_split=0.2,
                class_weight=class_weights,
                callbacks=callbacks,
                verbose=0
            )

            # Predict with threshold
            y_prob = model.predict(X_test).flatten()
            y_pred = (y_prob > 0.2).astype(int)

            cm = confusion_matrix(y_test, y_pred)
            acc, prec, rec, f1, macro_f1, fn_rate, pos_rate, cm_vals = compute_metrics(cm)
            confusion_sum += cm_vals
            fn_rates.append(fn_rate)
            pos_rates.append(pos_rate)
            accs.append(acc)
            precs.append(prec)
            recs.append(rec)
            f1s.append(f1)
            macro_f1s.append(macro_f1)

    avg_fn_rates.append(np.mean(fn_rates))
    avg_pos_rates.append(np.mean(pos_rates))
    all_metrics.append({
        "acc": np.mean(accs),
        "prec": np.mean(precs),
        "rec": np.mean(recs),
        "f1": np.mean(f1s),
        "macro_f1": np.mean(macro_f1s),
        "conf_matrix": (confusion_sum / (n_runs * 5)).round(2)
    })

# === Plot FN vs Pos Rate ===
plt.figure(figsize=(10, 6))
plt.plot(avg_pos_rates, avg_fn_rates, marker='o', label="NN Undersample")
plt.axhline(y=0.01, color='red', linestyle='--', label='FN Rate = 0.01')
plt.xlabel("Positive Rate")
plt.ylabel("False Negative Rate")
plt.title("NN: FN Rate vs Positive Rate across Undersample Ratios")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Find best ratio ===
results_df = pd.DataFrame({
    "Majority Ratio": [r[0] for r in ratios],
    "Minority Ratio": [r[1] for r in ratios],
    "Avg FN Rate": avg_fn_rates,
    "Avg Positive Rate": avg_pos_rates
})
filtered_df = results_df[results_df["Avg FN Rate"] <= 0.01].sort_values(by="Avg Positive Rate")

print("Best undersampling ratios with Avg FN Rate <= 0.01 and lowest Avg Positive Rate:")
print(filtered_df.head(10))

# === Final confusion matrix and metrics ===
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
