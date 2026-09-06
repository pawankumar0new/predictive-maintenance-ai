import joblib
import numpy as np
import pandas as pd
import os

# Get the project root directory (parent of src folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_model():
    """Load the trained model and feature names."""
    model_path = os.path.join(BASE_DIR, 'models', 'random_forest_model.pkl')
    features_path = os.path.join(BASE_DIR, 'models', 'feature_names.pkl')
    
    model = joblib.load(model_path)
    feature_names = joblib.load(features_path)
    return model, feature_names

def predict_failure(air_temp, process_temp, speed, torque, tool_wear):
    """
    Predict failure probability for given machine parameters.
    """
    # Load model
    model, feature_names = load_model()
    
    # Create input array with feature engineering
    temp_diff = process_temp - air_temp
    power = torque * speed * (2 * 3.14159 / 60)
    wear_torque = tool_wear * torque
    
    # Match feature order - use the saved feature names
    input_data = np.array([[air_temp, process_temp, speed, torque, tool_wear,
                            temp_diff, power, wear_torque]])
    
    # Get probability
    prob = model.predict_proba(input_data)[0][1]
    pred = model.predict(input_data)[0]
    
    # Risk level
    if prob < 0.3:
        risk_level = 'Normal'
    elif prob < 0.6:
        risk_level = 'Warning'
    else:
        risk_level = 'High Risk'
    
    health_score = int((1 - prob) * 100)
    
    return {
        'failure_probability': round(prob * 100, 2),
        'prediction': pred,
        'risk_level': risk_level,
        'health_score': health_score
    }

# Test it
if __name__ == "__main__":
    result = predict_failure(
        air_temp=300, 
        process_temp=310, 
        speed=1500, 
        torque=40, 
        tool_wear=50
    )
    print("Result:", result)