# test_rl.py  — paste this in root folder and run it
import numpy as np
import joblib
import tensorflow as tf

scaler = joblib.load('rl_agent_scaler.pkl')
model  = tf.keras.models.load_model('rl_agent.keras')

# Simulate a normal sample
normal = np.array([[15.0, 2400.0, 55.0, 52.0, 5.0, 14.0, 44.0]])
# Simulate a stress sample  
stress = np.array([[100.0, 3076.0, 58.0, 97.0, 0.0, 14.0, 48.0]])

print("=== WITHOUT scaler ===")
print(f"Normal : {model.predict(normal, verbose=0)[0][0]:.4f}")
print(f"Stress : {model.predict(stress, verbose=0)[0][0]:.4f}")

print("\n=== WITH scaler ===")
print(f"Normal : {model.predict(scaler.transform(normal), verbose=0)[0][0]:.4f}")
print(f"Stress : {model.predict(scaler.transform(stress), verbose=0)[0][0]:.4f}")