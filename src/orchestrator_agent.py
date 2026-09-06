from .agent_base import BaseAgent
from .data_agent import DataAgent
from .prediction_agent import PredictionAgent
from .explanation_agent import ExplanationAgent

class OrchestratorAgent(BaseAgent):
    """Main orchestrator that coordinates all agents."""
    
    def __init__(self):
        super().__init__(name="Orchestrator", role="System Coordinator")
        self.agents = {}
        self.history = []
        self._initialize_agents()
        
    def _initialize_agents(self):
        """Initialize all sub-agents."""
        self.log_activity("Initializing agents...")
        self.agents['data'] = DataAgent()
        self.agents['prediction'] = PredictionAgent()
        self.agents['explanation'] = ExplanationAgent()
        self.log_activity("All agents initialized successfully")
    
    def get_agent_status(self):
        """Get status of all agents."""
        return {
            name: agent.get_status()
            for name, agent in self.agents.items()
        }
    
    def process_prediction_request(self, features):
        """Process a complete prediction request through all agents."""
        self.status = "processing"
        self.log_activity(f"Processing prediction request: {features}")
        
        # Step 1: Validate input (Data Agent) - gets 3 return values now
        valid, message, converted_features = self.agents['data'].validate_input(features)
        if not valid:
            self.log_activity(f"Validation failed: {message}", "error")
            self.status = "idle"
            return {'error': message}
        
        # Step 2: Preprocess features using converted values
        processed_features = self.agents['data'].preprocess_features(converted_features)
        
        # Step 3: Get prediction (Prediction Agent)
        prediction = self.agents['prediction'].predict(processed_features)
        if prediction is None:
            self.log_activity("Prediction failed", "error")
            self.status = "idle"
            return {'error': "Prediction failed"}
        
        # Step 4: Get explanation (Explanation Agent)
        explanation = self.agents['explanation'].process({
            'probability': prediction['failure_probability'] / 100,
            'features': list(converted_features.values())  # Use converted values
        })
        
        # Step 5: Combine results
        result = {
            'status': 'success',
            'prediction': prediction,
            'explanation': explanation,
            'processed_features': processed_features,
            'timestamp': str(self.last_activity)
        }
        
        # Store in history
        self.history.append(result)
        self.log_activity("Prediction request completed successfully")
        self.status = "idle"
        
        return result
    
    def process(self, input_data):
        """Main processing method."""
        if 'features' in input_data:
            return self.process_prediction_request(input_data['features'])
        elif 'status' in input_data:
            return self.get_agent_status()
        elif 'history' in input_data:
            return self.history
        return {'error': "Invalid request"}