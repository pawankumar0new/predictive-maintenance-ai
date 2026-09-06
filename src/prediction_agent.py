import joblib
import numpy as np
import os
from .agent_base import BaseAgent

class PredictionAgent(BaseAgent):
    """Agent responsible for making predictions."""
    
    def __init__(self, model_path=None):
        super().__init__(name="PredictionAgent", role="Prediction Engine")
        self.model = None
        self.feature_names = None
        self.model_loaded = False
        self.model_path = model_path or self._get_default_path()
        
    def _get_default_path(self):
        """Get default model path."""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_dir, 'models', 'random_forest_model.pkl')
    
    def load_model(self):
        """Load the trained model."""
        self.log_activity(f"Loading model from {self.model_path}")
        try:
            self.model = joblib.load(self.model_path)
            self.model_loaded = True
            self.log_activity("Model loaded successfully")
            return True
        except Exception as e:
            self.log_activity(f"Error loading model: {e}", "error")
            self.model_loaded = False
            return False
    
    def predict(self, features):
        """Make a prediction."""
        if not self.model_loaded:
            if not self.load_model():
                return None
        
        self.status = "processing"
        self.log_activity(f"Making prediction for features: {features}")
        
        try:
            input_array = np.array([features])
            prob = self.model.predict_proba(input_array)[0][1]
            pred = self.model.predict(input_array)[0]
            
            result = {
                'failure_probability': round(prob * 100, 2),
                'prediction': int(pred),
                'confidence': round(max(prob, 1-prob) * 100, 2)
            }
            
            self.log_activity(f"Prediction result: {result}")
            self.status = "idle"
            return result
            
        except Exception as e:
            self.log_activity(f"Prediction error: {e}", "error")
            self.status = "idle"
            return None
    
    def process(self, input_data):
        """Main processing method."""
        if 'features' in input_data:
            return self.predict(input_data['features'])
        elif 'load' in input_data:
            return self.load_model()
        return None