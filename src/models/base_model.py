"""Base class for all ML models."""
import joblib
from pathlib import Path
from typing import Any, Dict
import numpy as np
import pandas as pd


class BaseModel:
    """Base class for insurance ML models."""
    
    def __init__(self, model_name: str, model_path: str = "data"):
        """Initialize base model.
        
        Args:
            model_name: Name of the model (e.g., 'fraud_detector')
            model_path: Directory where models are stored
        """
        self.model_name = model_name
        self.model_path = Path(model_path)
        self.model = None
        self.is_loaded = False
    
    def load(self) -> None:
        """Load trained model from disk."""
        model_file = self.model_path / f"{self.model_name}.pkl"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        self.model = joblib.load(model_file)
        self.is_loaded = True
        print(f"✅ {self.model_name} loaded successfully")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Make predictions.
        
        Args:
            data: Input features as DataFrame
            
        Returns:
            Predictions array
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model not loaded. Call load() first.")
        
        return self.model.predict(data)
    
    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities (for classifiers).
        
        Args:
            data: Input features as DataFrame
            
        Returns:
            Probability array
        """
        if not self.is_loaded:
            raise RuntimeError(f"Model not loaded. Call load() first.")
        
        if not hasattr(self.model, 'predict_proba'):
            raise AttributeError(f"{self.model_name} doesn't support probabilities")
        
        return self.model.predict_proba(data)
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information.
        
        Returns:
            Dictionary with model metadata
        """
        return {
            "model_name": self.model_name,
            "is_loaded": self.is_loaded,
            "model_type": type(self.model).__name__ if self.model else None
        }