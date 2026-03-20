import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import joblib
import os

# -----------------------------
# Step 1: Load Dataset
# -----------------------------
df = pd.read_csv('data/study_data.csv')
df.columns = df.columns.str.strip()  # Clean column names

print(f"Dataset shape: {df.shape}")
print("First 5 rows:\n", df.head())

# -----------------------------
# Step 2: Prepare Features & Target
# -----------------------------
X = df.drop(columns=["Productivity Score", "Burnout Risk"])
y = df["Productivity Score"]

print("\nFeatures:", list(X.columns))
print("Target: Productivity Score")

# -----------------------------
# Step 3: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining samples: {len(X_train)}, Testing samples: {len(X_test)}")

# -----------------------------
# Step 4: Train Linear Regression Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training complete!")

# -----------------------------
# Step 5: Make Predictions
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# Step 6: Evaluate Model
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "="*40)
print("TEST SET EVALUATION")
print("="*40)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("\nCross-Validation R² Scores:", cv_scores)
print(f"Average CV R²: {np.mean(cv_scores):.4f}")
print(f"Std Deviation CV R²: {np.std(cv_scores):.4f}")

# -----------------------------
# Step 7: Feature Importance (Coefficients)
# -----------------------------
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\n" + "="*40)
print("FEATURE IMPORTANCE")
print("="*40)
print(coefficients.to_string(index=False))

# Optional: Visualize coefficients
# Commented out to prevent blocking
# plt.figure(figsize=(8,5))
# plt.bar(coefficients['Feature'], coefficients['Coefficient'])
# plt.title("Feature Importance (Linear Regression Coefficients)")
# plt.xlabel("Feature")
# plt.ylabel("Coefficient Value")
# plt.xticks(rotation=45)
# plt.tight_layout()
# plt.show()

# -----------------------------
# Step 8: Save Model for API
# -----------------------------
os.makedirs('models', exist_ok=True)

# Save the trained model
model_path = 'models/productivity_model.pkl'
joblib.dump(model, model_path)
print(f"\n✓ Model saved to: {model_path}")

# Save feature names for validation
feature_path = 'models/productivity_features.pkl'
joblib.dump(list(X.columns), feature_path)
print(f"✓ Feature names saved to: {feature_path}")

print("\nTRAINING COMPLETE! ✓")
