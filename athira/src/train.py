import os
import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from features import extract_features

def train_model():
    data_dir = os.path.join("data", "train")
    X = []
    y = []
    
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    if not classes:
        print("No data found in data/train. Run generate_data.py first.")
        return

    print(f"Found classes: {classes}")

    for label in classes:
        class_dir = os.path.join(data_dir, label)
        for filename in os.listdir(class_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(class_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()
                    feats = extract_features(text)
                    X.append(feats)
                    y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Training on {len(X)} samples...")
    
    # Create a pipeline with scaling and SVM
    # Scaling is crucial for SVM performance
    model = make_pipeline(StandardScaler(), SVC(kernel='linear', probability=True))
    model.fit(X, y)
    
    # Save model
    model_path = os.path.join("models", "author_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_model()
