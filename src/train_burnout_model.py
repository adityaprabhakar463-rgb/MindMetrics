import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score
import joblib
import os

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv("data/study_data.csv")
df.columns = df.columns.str.strip()

X = df.drop(columns=["Productivity Score", "Burnout Risk"])
y = df["Burnout Risk"]

# -----------------------------
# Step 2: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Step 3: Logistic Regression Model
# -----------------------------
model = LogisticRegression(max_iter=1000, class_weight='balanced')
model.fit(X_train, y_train)

# -----------------------------
# Step 4: Predict Probabilities
# -----------------------------
y_prob = model.predict_proba(X_test)[:,1]  # probability of burnout

# -----------------------------
# Step 5: Evaluate Multiple Thresholds
# -----------------------------
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

print("\nThreshold Analysis:\n")
for t in thresholds:
    y_pred = (y_prob >= t).astype(int)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"Threshold: {t:.2f} | Accuracy: {acc:.4f} | Recall (Burnout): {recall:.2f}")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("-"*50)

# Optional: show classification report for chosen threshold
chosen_threshold = 0.3
y_pred_final = (y_prob >= chosen_threshold).astype(int)
print("\nClassification Report (Threshold = 0.3):\n", classification_report(y_test, y_pred_final))

# -----------------------------
# Step 6: Save Model for API
# -----------------------------
os.makedirs('models', exist_ok=True)

# Save the trained model
model_path = 'models/burnout_model.pkl'
joblib.dump(model, model_path)
print(f"\n✓ Model saved to: {model_path}")

# Save optimal threshold
threshold_path = 'models/burnout_threshold.pkl'
joblib.dump(chosen_threshold, threshold_path)
print(f"✓ Threshold saved to: {threshold_path}")

# Save feature names for validation
feature_path = 'models/burnout_features.pkl'
joblib.dump(list(X.columns), feature_path)
print(f"✓ Feature names saved to: {feature_path}")

print("\nTRAINING COMPLETE! ✓")

