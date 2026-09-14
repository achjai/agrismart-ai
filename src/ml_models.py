import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import os

def create_synthetic_crop_data():
    """Generates a synthetic dataset for crop recommendation mimicking the Kaggle dataset."""
    crops = ['rice', 'maize', 'chickpea', 'kidneybeans', 'pigeonpeas',
             'mothbeans', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
             'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple',
             'orange', 'papaya', 'coconut', 'cotton', 'jute', 'coffee']
    
    data = []
    np.random.seed(42)
    
    for crop in crops:
        # Generate 100 samples per crop with some random distribution
        for _ in range(100):
            # These ranges are roughly based on agronomic requirements
            if crop == 'rice':
                n, p, k, temp, hum, ph, rain = np.random.normal(80, 10), np.random.normal(45, 10), np.random.normal(40, 10), np.random.normal(25, 3), np.random.normal(82, 5), np.random.normal(6.5, 0.5), np.random.normal(200, 30)
            elif crop == 'wheat':
                n, p, k, temp, hum, ph, rain = np.random.normal(90, 10), np.random.normal(50, 10), np.random.normal(45, 10), np.random.normal(20, 3), np.random.normal(60, 5), np.random.normal(6.5, 0.5), np.random.normal(100, 20)
            elif crop == 'cotton':
                n, p, k, temp, hum, ph, rain = np.random.normal(120, 10), np.random.normal(45, 10), np.random.normal(45, 10), np.random.normal(25, 3), np.random.normal(80, 5), np.random.normal(6.5, 0.5), np.random.normal(80, 20)
            elif crop == 'coffee':
                n, p, k, temp, hum, ph, rain = np.random.normal(100, 10), np.random.normal(25, 5), np.random.normal(30, 5), np.random.normal(25, 3), np.random.normal(60, 5), np.random.normal(6.5, 0.5), np.random.normal(150, 30)
            elif crop == 'apple':
                n, p, k, temp, hum, ph, rain = np.random.normal(20, 10), np.random.normal(130, 10), np.random.normal(200, 20), np.random.normal(22, 2), np.random.normal(92, 2), np.random.normal(5.9, 0.3), np.random.normal(110, 10)
            else:
                # Generic fallback with random centers for other crops
                n = np.random.normal(np.random.randint(10, 120), 10)
                p = np.random.normal(np.random.randint(10, 120), 10)
                k = np.random.normal(np.random.randint(10, 120), 10)
                temp = np.random.normal(np.random.randint(15, 35), 3)
                hum = np.random.normal(np.random.randint(40, 95), 5)
                ph = np.random.normal(np.random.randint(5, 8), 0.5)
                rain = np.random.normal(np.random.randint(40, 250), 30)
                
            data.append([
                max(0, n), max(0, p), max(0, k), 
                max(0, temp), max(0, min(100, hum)), 
                max(0, min(14, ph)), max(0, rain), 
                crop
            ])
            
    df = pd.DataFrame(data, columns=['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label'])
    return df

def train_and_save_model():
    print("Generating synthetic crop dataset...")
    df = create_synthetic_crop_data()
    
    X = df.drop('label', axis=1)
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model trained successfully. Test Accuracy: {accuracy:.4f}")
    
    # Save the model
    os.makedirs('../model/weights', exist_ok=True)
    model_path = '../model/weights/crop_recommendation.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
