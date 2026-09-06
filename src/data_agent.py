import pandas as pd
import numpy as np
from .agent_base import BaseAgent

class DataAgent(BaseAgent):
    """Agent responsible for data loading, validation, and preprocessing."""
    
    def __init__(self):
        super().__init__(name="DataAgent", role="Data Management")
        self.dataset = None
        self.feature_columns = None
        self.target_column = None
        
    def load_data(self, filepath):
        """Load the dataset."""
        self.log_activity(f"Loading data from {filepath}")
        try:
            self.dataset = pd.read_csv(filepath)
            self.log_activity(f"Loaded {len(self.dataset)} records")
            return self.dataset
        except Exception as e:
            self.log_activity(f"Error loading data: {e}", "error")
            raise
    
    def validate_input(self, features):
        """Validate incoming prediction request and convert to float."""
        required_fields = [
            'air_temp', 'process_temp', 'speed', 'torque', 'tool_wear'
        ]
        
        self.log_activity(f"Validating input: {features}")
        
        # Check for missing fields
        missing = [f for f in required_fields if f not in features]
        if missing:
            error = f"Missing required fields: {missing}"
            self.log_activity(error, "error")
            return False, error, None
        
        # Convert all values to float and validate ranges
        validations = {
            'air_temp': (250, 350),
            'process_temp': (250, 400),
            'speed': (1000, 3000),
            'torque': (0, 100),
            'tool_wear': (0, 300)
        }
        
        converted_features = {}
        
        for field, (min_val, max_val) in validations.items():
            try:
                # Convert to float (handles int, float, and string numbers)
                value = float(features[field])
                converted_features[field] = value
                
                if value < min_val or value > max_val:
                    error = f"{field} ({value}) must be between {min_val} and {max_val}"
                    self.log_activity(error, "error")
                    return False, error, None
                    
            except (ValueError, TypeError) as e:
                error = f"{field} must be a number (got: {features[field]})"
                self.log_activity(error, "error")
                return False, error, None
        
        self.log_activity("Input validation successful")
        return True, "Valid", converted_features
    
    def preprocess_features(self, features):
        """Engineer additional features."""
        self.log_activity("Engineering features")
        
        air_temp = features['air_temp']
        process_temp = features['process_temp']
        speed = features['speed']
        torque = features['torque']
        tool_wear = features['tool_wear']
        
        # Feature engineering
        temp_diff = process_temp - air_temp
        power = torque * speed * (2 * np.pi / 60)
        wear_torque = tool_wear * torque
        
        processed = [
            air_temp, process_temp, speed, torque, tool_wear,
            temp_diff, power, wear_torque
        ]
        
        self.log_activity(f"Engineered features: {processed}")
        return processed
    
    def process(self, input_data):
        """Main processing method."""
        self.status = "processing"
        
        if 'filepath' in input_data:
            result = self.load_data(input_data['filepath'])
        elif 'validate' in input_data:
            valid, message, converted = self.validate_input(input_data['features'])
            result = {'valid': valid, 'message': message, 'converted_features': converted}
        elif 'features' in input_data:
            result = self.preprocess_features(input_data['features'])
        else:
            result = None
        
        self.status = "idle"
        return result