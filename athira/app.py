import sys
import os
import joblib
import numpy as np
from flask import Flask, render_template, request

# Add src to path to import features module
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from features import extract_features

app = Flask(__name__)

# Load Model
MODEL_PATH = os.path.join("models", "author_model.pkl")
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("WARNING: Model not found. Please run src/train.py first.")

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction=None)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return render_template('index.html', prediction="Error: Model not loaded.", confidence=0, features=[0]*5)

    text = request.form['text']
    
    # Extract features
    features = extract_features(text)
    
    # Reshape for prediction (single sample)
    features_reshaped = features.reshape(1, -1)
    
    # Predict
    prediction_cls = model.predict(features_reshaped)[0]
    probabilities = model.predict_proba(features_reshaped)[0]
    
    # Get confidence score (max probability)
    confidence = np.max(probabilities) * 100
    
    return render_template('index.html', 
                           original_text=text, 
                           prediction=prediction_cls, 
                           confidence=round(confidence, 2),
                           features=features)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
