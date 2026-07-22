"""
AI Virtual Doctor - Model Training
LinearSVC + Probability Calibration
Compatible with latest scikit-learn
"""

import os
import pickle
import pandas as pd
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET = os.path.join(
    BASE_DIR,
    "dataset",
    "Diseases_and_Symptoms_dataset.csv"
)

MODEL_OUT = os.path.join(BASE_DIR, "model", "disease_model.pkl")
ENCODER_OUT = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
SYMPTOM_OUT = os.path.join(BASE_DIR, "model", "symptoms_list.pkl")

os.makedirs(os.path.join(BASE_DIR, "model"), exist_ok=True)

# -----------------------------
# Load Dataset
# -----------------------------
print("Loading dataset...")

df = pd.read_csv(DATASET)

df["diseases"] = df["diseases"].str.strip()

# -----------------------------
# Features & Labels
# -----------------------------
symptom_cols = [c for c in df.columns if c != "diseases"]

X = df[symptom_cols].astype(np.float32).values
y = df["diseases"].values

# Save symptom list
with open(SYMPTOM_OUT, "wb") as f:
    pickle.dump(symptom_cols, f)

# Encode labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

with open(ENCODER_OUT, "wb") as f:
    pickle.dump(encoder, f)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.10,
    random_state=42,
    stratify=y_encoded
)

print(f"Training Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# -----------------------------
# Build Model
# -----------------------------
print("\nTraining LinearSVC...")

base_model = LinearSVC(
    C=2.0,
    max_iter=5000,
    random_state=42
)

model = CalibratedClassifierCV(
    estimator=base_model,
    cv=5
)

# -----------------------------
# Train
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# Evaluate
# -----------------------------
pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\n===============================")
print(f"Accuracy : {accuracy*100:.2f}%")
print("===============================\n")

# -----------------------------
# Save Model
# -----------------------------
with open(MODEL_OUT, "wb") as f:
    pickle.dump(model, f)

print("Model Saved Successfully!")
print(MODEL_OUT)

print("\nTraining Completed Successfully.")
