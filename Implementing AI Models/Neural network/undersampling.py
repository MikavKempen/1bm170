import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve
)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Load data
df = pd.read_csv("heatpump_nn_ready.csv")
X = df.drop(columns=["Target_Broken"])
y = df["Target_Broken"]

# Fill missing values
X = X.fillna(0)

# Split before scaling and SMOTE
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.3, random_state=42
)

# Scale
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----- Feature Selection -----
# Use RandomForest for feature importance
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
rf.fit(X_train_scaled, y_train)

# Select important features
selector = SelectFromModel(rf, prefit=True, threshold='median')  # Top 50% features
X_train_selected = selector.transform(X_train_scaled)
X_test_selected = selector.transform(X_test_scaled)

# Apply 67/33 undersampling on training data
X_y_train = pd.DataFrame(X_train_selected)
X_y_train["Target_Broken"] = y_train.values

# Separate classes
minority = X_y_train[X_y_train["Target_Broken"] == 1]
majority = X_y_train[X_y_train["Target_Broken"] == 0]

# Undersample majority: keep 33%
majority_downsampled = majority.sample(frac=0.33, random_state=42)

# Combine
undersampled = pd.concat([minority, majority_downsampled])
X_train_sm = undersampled.drop(columns="Target_Broken").values
y_train_sm = undersampled["Target_Broken"].values


# Class weights (on original y_train)
weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
class_weights = dict(enumerate(weights))

# Focal Loss for class imbalance
def focal_loss(gamma=2., alpha=0.25):
    def loss(y_true, y_pred):
        bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        return alpha * tf.keras.backend.pow(1. - pt, gamma) * bce
    return loss


# Model
model = Sequential([
    Dense(256, input_shape=(X_train_sm.shape[1],)),
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

# Callbacks
callbacks = [
    EarlyStopping(patience=12, restore_best_weights=True),
    ReduceLROnPlateau(patience=5, factor=0.5, min_lr=1e-5)
]

# Train
model.fit(
    X_train_sm, y_train_sm,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=1
)

# Predict
y_prob = model.predict(X_test_selected).flatten()
threshold = 0.2
y_pred = (y_prob > threshold).astype(int)

# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred, zero_division=0))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC AUC:", roc_auc_score(y_test, y_prob))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label='ROC curve')
plt.plot([0, 1], [0, 1], linestyle='--', label='Random baseline')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()
