"""
train_rl.py
===========
Trains the neural-network classifier (RL agent).
Run directly: python train_rl.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

DATA_PATH   = Path('data/labeled_training.csv')
MODEL_PATH  = Path('rl_agent.keras')
SCALER_PATH = Path('rl_agent_scaler.pkl')

FEATURES = [
    'cpu_percent', 'cpu_freq', 'cpu_memory', 'cpu_temp',
    'gpu_percent', 'gpu_memory', 'gpu_temp',
]

# ── Load data ─────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)

# Label convention: 0 = normal, 1 = anomaly (from collect_labeled.py)
X = df[FEATURES].values
y = df['label'].values.astype(int)   # DO NOT flip! 0 = normal, 1 = anomaly

print(f"Dataset: {len(df)} samples | normal={sum(y==0)} | anomaly={sum(y==1)}")

if sum(y==1) == 0:
    print("ERROR: No anomaly samples found in labeled_training.csv")
    print("       Run collect_labeled.py and press 'a' during a stress test.")
    exit(1)

# ── Scale features ────────────────────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, SCALER_PATH)
print(f"Scaler saved → {SCALER_PATH}")

# ── Train / val split ─────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ── Class weights ─────────────────────────────────────────────────────────────
classes      = np.unique(y_train)
cw           = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight = dict(zip(classes.tolist(), cw.tolist()))
print(f"Class weights: {class_weight}")

# ── Model ─────────────────────────────────────────────────────────────────────
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(len(FEATURES),)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')],
)

model.summary()

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_auc', patience=5,
        restore_best_weights=True, mode='max'
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
    ),
]

# ── Train ─────────────────────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1,
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
val_loss, val_acc, val_auc = model.evaluate(X_val, y_val, verbose=0)
print(f"\nValidation — loss={val_loss:.4f}  acc={val_acc:.4f}  auc={val_auc:.4f}")

# ── Quick test ────────────────────────────────────────────────────────────────
sample = X_scaled[0].reshape(1, -1)
pred_sample = model.predict(sample, verbose=0)[0, 0]
print(f"Sample prediction probability: {pred_sample:.4f}")

# ── Save ──────────────────────────────────────────────────────────────────────
model.save(MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")
print(f"Scaler saved → {SCALER_PATH}")
print("\nDone. Now restart monitor.py.")