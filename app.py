"""
Flask API for Student Productivity Predictor
Serves predictions from trained ML models for productivity score and burnout risk.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os

app = Flask(__name__)

# Enable CORS for frontend communication (allow all origins for dev)
CORS(app, resources={r"/api/*": {"origins": "*"}})


# Load models at startup
print("Loading models...")

try:
    # Load productivity model
    productivity_model = joblib.load('models/productivity_model.pkl')
    productivity_features = joblib.load('models/productivity_features.pkl')
    print("✓ Productivity model loaded")
    
    # Load burnout model
    burnout_model = joblib.load('models/burnout_model.pkl')
    burnout_threshold = joblib.load('models/burnout_threshold.pkl')
    burnout_features = joblib.load('models/burnout_features.pkl')
    print("✓ Burnout model loaded")
    
    models_loaded = True
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please train the models first by running:")
    print("  python src/train_model.py")
    print("  python src/train_burnout_model.py")
    models_loaded = False

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        "message": "Student Productivity Predictor API",
        "status": "running",
        "models_loaded": models_loaded,
        "endpoints": {
            "productivity": "/api/predict/productivity",
            "burnout": "/api/predict/burnout"
        }
    })

@app.route('/api/predict/productivity', methods=['POST'])
def predict_productivity():
    """
    Predict productivity score
    Expected input JSON:
    {
        "Study Hours": 6.5,
        "Sleep Hours": 7.0,
        "Mood": 4,
        "Distraction": 2,
        "Difficulty": 3,
        "Exam Proximity": 15
    }
    """
    if not models_loaded:
        return jsonify({"error": "Models not loaded. Please train models first."}), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        # Validate input
        missing_features = [f for f in productivity_features if f not in data]
        if missing_features:
            return jsonify({
                "error": f"Missing required features: {missing_features}",
                "required_features": productivity_features
            }), 400
        
        # Create feature array in correct order
        features = np.array([[data[f] for f in productivity_features]])
        
        # Make prediction
        prediction = float(productivity_model.predict(features)[0])
        
        # Clip to valid range (0-100)
        prediction = max(0, min(100, prediction))
        
        return jsonify({
            "productivity_score": round(prediction, 2),
            "input": data,
            "interpretation": get_productivity_interpretation(prediction)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict/burnout', methods=['POST'])
def predict_burnout():
    """
    Predict burnout risk
    Expected input JSON:
    {
        "Study Hours": 6.5,
        "Sleep Hours": 7.0,
        "Mood": 4,
        "Distraction": 2,
        "Difficulty": 3,
        "Exam Proximity": 15
    }
    """
    if not models_loaded:
        return jsonify({"error": "Models not loaded. Please train models first."}), 500
    
    try:
        # Get input data
        data = request.get_json()
        
        # Validate input
        missing_features = [f for f in burnout_features if f not in data]
        if missing_features:
            return jsonify({
                "error": f"Missing required features: {missing_features}",
                "required_features": burnout_features
            }), 400
        
        # Create feature array in correct order
        features = np.array([[data[f] for f in burnout_features]])
        
        # Get probability of burnout
        probability = float(burnout_model.predict_proba(features)[0][1])
        
        # Apply threshold
        burnout_risk = int(probability >= burnout_threshold)
        
        return jsonify({
            "burnout_risk": burnout_risk,
            "burnout_probability": round(probability, 4),
            "threshold": burnout_threshold,
            "input": data,
            "interpretation": get_burnout_interpretation(burnout_risk, probability)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def get_productivity_interpretation(score):
    """Return interpretation of productivity score"""
    if score >= 80:
        return "Excellent productivity! 🌟"
    elif score >= 60:
        return "Good productivity. Keep it up! 👍"
    elif score >= 40:
        return "Moderate productivity. Room for improvement. 📚"
    else:
        return "Low productivity. Consider adjusting study habits. ⚠️"

def get_burnout_interpretation(risk, probability):
    """Return interpretation of burnout risk"""
    if risk == 1:
        return f"High burnout risk detected ({probability*100:.1f}% probability). Consider taking breaks and reducing stress. ⚠️"
    else:
        return f"Low burnout risk ({probability*100:.1f}% probability). Maintain healthy study habits! ✓"

if __name__ == '__main__':
    print("\n" + "="*60)
    print("Starting Flask API Server...")
    print("="*60)
    if models_loaded:
        print("✓ All models loaded successfully")
        print("\nAPI Endpoints:")
        print("  - GET  /                          - API info")
        print("  - POST /api/predict/productivity  - Predict productivity score")
        print("  - POST /api/predict/burnout       - Predict burnout risk")
    else:
        print("⚠ Models not loaded. Train them first!")
    print("\nServer running at: http://localhost:5001")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5001, host='0.0.0.0')

