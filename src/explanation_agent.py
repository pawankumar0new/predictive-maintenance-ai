import numpy as np
from .agent_base import BaseAgent

class ExplanationAgent(BaseAgent):
    """Agent responsible for explaining predictions and providing recommendations."""
    
    def __init__(self):
        super().__init__(name="ExplanationAgent", role="Decision Explanation")
        
        # Define thresholds and explanations
        self.thresholds = {
            'air_temp': {'high': 320, 'low': 280, 'unit': 'K'},
            'process_temp': {'high': 340, 'low': 290, 'unit': 'K'},
            'speed': {'high': 2500, 'low': 1200, 'unit': 'rpm'},
            'torque': {'high': 60, 'low': 10, 'unit': 'Nm'},
            'tool_wear': {'high': 200, 'low': 50, 'unit': 'min'}
        }
        
        self.risk_factors = {
            'air_temp': "Air temperature is {status}",
            'process_temp': "Process temperature is {status}",
            'speed': "Rotational speed is {status}",
            'torque': "Torque level is {status}",
            'tool_wear': "Tool wear is {status}"
        }
        
        self.recommendations = {
            'Normal': {
                'message': "✅ Machine is operating normally. Continue monitoring.",
                'action': 'monitor'
            },
            'Warning': {
                'message': "⚠️ Schedule inspection soon. Monitor key parameters closely.",
                'action': 'inspect'
            },
            'High Risk': {
                'message': "🚨 High risk detected! Consider immediate maintenance action.",
                'action': 'maintenance'
            }
        }
    
    def calculate_health_score(self, probability):
        """Calculate health score from failure probability."""
        return int((1 - probability) * 100)
    
    def determine_risk_level(self, probability):
        """Determine risk level from probability."""
        if probability < 0.3:
            return 'Normal'
        elif probability < 0.6:
            return 'Warning'
        else:
            return 'High Risk'
    
    def identify_risk_factors(self, features):
        """Identify which features contributed to the risk."""
        risk_factors = []
        
        for i, (name, values) in enumerate(self.thresholds.items()):
            try:
                value = features[i]
                if value > values['high']:
                    risk_factors.append(f"{name}: High ({value} {values['unit']})")
                elif value < values['low']:
                    risk_factors.append(f"{name}: Low ({value} {values['unit']})")
            except (IndexError, KeyError):
                pass
        
        return risk_factors
    
    def generate_recommendation(self, probability, features=None):
        """Generate a maintenance recommendation."""
        risk_level = self.determine_risk_level(probability)
        health_score = self.calculate_health_score(probability)
        
        recommendation = self.recommendations[risk_level]
        
        # Add additional context if risk factors are available
        if features and probability > 0.3:
            risk_factors = self.identify_risk_factors(features)
            if risk_factors:
                recommendation['details'] = f"Risk factors: {', '.join(risk_factors)}"
        
        return {
            'risk_level': risk_level,
            'health_score': health_score,
            'message': recommendation['message'],
            'action': recommendation['action'],
            'risk_factors': self.identify_risk_factors(features) if features else []
        }
    
    def process(self, input_data):
        """Main processing method."""
        self.status = "processing"
        
        probability = input_data.get('probability', 0)
        features = input_data.get('features', None)
        
        result = self.generate_recommendation(probability, features)
        
        self.log_activity(f"Generated recommendation: {result['risk_level']}")
        self.status = "idle"
        return result
    