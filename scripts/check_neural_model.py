"""
check_neural_model.py
=====================
Manual sanity check for the trained neural detector: confirms the saved model
and its scaler agree, and that scaling actually matters.

    python scripts/check_neural_model.py

This used to live at detectors/test.py, where two things went wrong: the name
made pytest collect it as a test module and execute the whole script (loading
TensorFlow) during collection, and its relative paths only resolved when it
was run from the repository root.
"""

import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from console import enable_unicode_output

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / 'neural_detector.keras'
SCALER_PATH = BASE_DIR / 'neural_detector_scaler.pkl'

# cpu_percent, cpu_freq, cpu_memory, cpu_temp, gpu_percent, gpu_memory, gpu_temp
NORMAL = np.array([[15.0, 2400.0, 55.0, 52.0, 5.0, 14.0, 44.0]])
STRESS = np.array([[100.0, 3076.0, 58.0, 97.0, 0.0, 14.0, 48.0]])


def main() -> int:
    enable_unicode_output()
    for path in (MODEL_PATH, SCALER_PATH):
        if not path.exists():
            print(f"ERROR: {path} not found — run train_neural.py first.")
            return 1

    import tensorflow as tf
    scaler = joblib.load(SCALER_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)

    def p(x):
        return float(model.predict(x, verbose=0)[0][0])

    print("=== WITHOUT scaler (wrong: the model was fitted on scaled input) ===")
    print(f"Normal : {p(NORMAL):.4f}")
    print(f"Stress : {p(STRESS):.4f}")

    print("\n=== WITH scaler (what NeuralDetector does) ===")
    normal_p, stress_p = p(scaler.transform(NORMAL)), p(scaler.transform(STRESS))
    print(f"Normal : {normal_p:.4f}")
    print(f"Stress : {stress_p:.4f}")

    # The head outputs P(anomaly), so the stress sample must score higher.
    ok = stress_p > normal_p
    print(f"\n{'PASS' if ok else 'FAIL'}: P(anomaly) is "
          f"{'higher' if ok else 'NOT higher'} for the stress sample.")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
