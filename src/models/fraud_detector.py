"""Fraud detection model for insurance claims."""
import pandas as pd
import numpy as np
from typing import Dict, Any
from .base_model import BaseModel


class FraudDetector(BaseModel):
    """Fraud detection model for insurance claims."""
    
    def __init__(self, model_path: str = "data"):
        """Initialize fraud detector.
        
        Args:
            model_path: Directory where models are stored
        """
        super().__init__("fraud_detector", model_path)
        self.threshold = 0.5  # Default fraud threshold
    
    def predict_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict fraud for a single claim.
        
        Args:
            claim_data: Dictionary with claim features:
                - claim_amount: float
                - days_to_report: int
                - claimant_age: int
                - prior_claims: int
        
        Returns:
            Dictionary with prediction results
        """
        # Convert to DataFrame
        df = pd.DataFrame([claim_data])
        
        # Ensure correct column order
        feature_cols = ['claim_amount', 'days_to_report', 'claimant_age', 'prior_claims']
        df = df[feature_cols]
        
        # Get prediction and probability
        prediction = self.predict(df)[0]
        probabilities = self.predict_proba(df)[0]
        fraud_probability = probabilities[1]  # Probability of fraud class
        
        # Determine confidence
        if fraud_probability > 0.8 or fraud_probability < 0.2:
            confidence = "high"
        elif fraud_probability > 0.6 or fraud_probability < 0.4:
            confidence = "medium"
        else:
            confidence = "low"
        
        # Determine action
        if fraud_probability > self.threshold:
            action = "REFER_TO_SIU"  # Special Investigation Unit
            risk_level = "HIGH"
        elif fraud_probability > 0.3:
            action = "ENHANCED_REVIEW"
            risk_level = "MEDIUM"
        else:
            action = "NORMAL_PROCESSING"
            risk_level = "LOW"
        
        return {
            "prediction": "fraud" if prediction == 1 else "legitimate",
            "fraud_probability": float(fraud_probability),
            "confidence": confidence,
            "risk_level": risk_level,
            "recommended_action": action,
            "claim_data": claim_data
        }
    
    def set_threshold(self, threshold: float) -> None:
        """Set custom fraud threshold.
        
        Args:
            threshold: New threshold (0.0 to 1.0)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = threshold
        print(f"✅ Fraud threshold updated to {threshold}")