from abc import ABC, abstractmethod
import logging
from datetime import datetime
import json

class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""
    
    def __init__(self, name, role):
        self.name = name
        self.role = role
        self.logger = self._setup_logger()
        self.status = "idle"
        self.last_activity = None
        
    def _setup_logger(self):
        logger = logging.getLogger(f"Agent_{self.name}")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        return logger
    
    @abstractmethod
    def process(self, input_data):
        """Main processing method - must be implemented by subclasses."""
        pass
    
    def log_activity(self, message, level="info"):
        """Log agent activity."""
        self.last_activity = datetime.now()
        if level == "info":
            self.logger.info(message)
        elif level == "warning":
            self.logger.warning(message)
        elif level == "error":
            self.logger.error(message)
    
    def get_status(self):
        return {
            "name": self.name,
            "role": self.role,
            "status": self.status,
            "last_activity": str(self.last_activity)
        }