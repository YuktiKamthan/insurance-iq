"""Fraud detection model for insurance claims."""
import pandas as pd
import numpy as np
from typing import Dict, Any
import shap
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
    
    def explain_prediction(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get SHAP explanation for a fraud prediction.
        
        Args:
            claim_data: Dictionary with claim features
        
        Returns:
            Dictionary with prediction and explanation
        """
        # Get basic prediction first
        result = self.predict_claim(claim_data)
        
        # Convert to DataFrame
        df = pd.DataFrame([claim_data])
        feature_cols = ['claim_amount', 'days_to_report', 'claimant_age', 'prior_claims']
        X = df[feature_cols]
        
        try:
            # Create SHAP explainer
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X)
            
            # Get feature contributions for fraud class (class 1)
            # Handle different SHAP output formats
            if isinstance(shap_values, list) and len(shap_values) == 2:
                # Binary classification with separate arrays for each class
                contributions = shap_values[1][0]
            elif isinstance(shap_values, np.ndarray):
                # Single array output
                if len(shap_values.shape) == 2:
                    contributions = shap_values[0]
                else:
                    contributions = shap_values
            else:
                contributions = shap_values[0]
            
            # Create explanation
            explanations = []
            for i, feature in enumerate(feature_cols):
                # Safely convert contribution to float
                if isinstance(contributions[i], np.ndarray):
                    contribution = float(contributions[i].item())
                else:
                    contribution = float(contributions[i])
                    
                value = float(X.iloc[0][feature])
                
                # Determine impact
                if abs(contribution) > 0.1:
                    impact = "high"
                elif abs(contribution) > 0.05:
                    impact = "medium"
                else:
                    impact = "low"
                
                # Determine direction
                direction = "increases" if contribution > 0 else "decreases"
                
                explanations.append({
                    "feature": feature,
                    "value": value,
                    "contribution": contribution,
                    "impact": impact,
                    "direction": direction
                })
            
            # Sort by absolute contribution
            explanations.sort(key=lambda x: abs(x['contribution']), reverse=True)
            
            # Add to result
            result["explanation"] = {
                "top_factors": explanations[:3],  # Top 3 contributors
                "all_factors": explanations
            }
            
        except Exception as e:
            # If SHAP fails, still return the prediction without explanation
            result["explanation"] = {
                "error": f"SHAP explanation failed: {str(e)}",
                "top_factors": [],
                "all_factors": []
            }
        
        return result

 
    
    def set_threshold(self, threshold: float) -> None:
        """Set custom fraud threshold.
        
        Args:
            threshold: New threshold (0.0 to 1.0)
        """
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        
        self.threshold = threshold
        print(f"✅ Fraud threshold updated to {threshold}")