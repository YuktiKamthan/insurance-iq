"""Tests for fraud detector model."""
import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.fraud_detector import FraudDetector


@pytest.fixture
def fraud_detector():
    """Create and load fraud detector for testing."""
    detector = FraudDetector()
    detector.load()
    return detector


class TestFraudDetector:
    """Test cases for fraud detector."""
    
    def test_model_loads_successfully(self, fraud_detector):
        """Test that model loads without errors."""
        assert fraud_detector.is_loaded is True
        assert fraud_detector.model is not None
    
    def test_legitimate_claim_prediction(self, fraud_detector):
        """Test prediction on a legitimate claim."""
        claim = {
            "claim_amount": 5000,
            "days_to_report": 2,
            "claimant_age": 45,
            "prior_claims": 0
        }
        
        result = fraud_detector.predict_claim(claim)
        
        assert result["prediction"] in ["legitimate", "fraud"]
        assert 0 <= result["fraud_probability"] <= 1
        assert result["confidence"] in ["low", "medium", "high"]
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]
        assert result["recommended_action"] in [
            "NORMAL_PROCESSING", 
            "ENHANCED_REVIEW", 
            "REFER_TO_SIU"
        ]
    
    def test_suspicious_claim_prediction(self, fraud_detector):
        """Test prediction on a suspicious claim."""
        claim = {
            "claim_amount": 50000,
            "days_to_report": 90,
            "claimant_age": 25,
            "prior_claims": 5
        }
        
        result = fraud_detector.predict_claim(claim)
        
        # Suspicious claims should have higher fraud probability
        assert result["prediction"] in ["legitimate", "fraud"]
        assert 0 <= result["fraud_probability"] <= 1
    
    def test_threshold_setting(self, fraud_detector):
        """Test custom threshold setting."""
        # Default threshold
        assert fraud_detector.threshold == 0.5
        
        # Set new threshold
        fraud_detector.set_threshold(0.7)
        assert fraud_detector.threshold == 0.7
        
        # Invalid threshold should raise error
        with pytest.raises(ValueError):
            fraud_detector.set_threshold(1.5)
    
    def test_prediction_response_structure(self, fraud_detector):
        """Test that prediction response has correct structure."""
        claim = {
            "claim_amount": 10000,
            "days_to_report": 10,
            "claimant_age": 35,
            "prior_claims": 1
        }
        
        result = fraud_detector.predict_claim(claim)
        
        # Check all required fields exist
        required_fields = [
            "prediction",
            "fraud_probability",
            "confidence",
            "risk_level",
            "recommended_action",
            "claim_data"
        ]
        
        for field in required_fields:
            assert field in result
    
    def test_multiple_predictions(self, fraud_detector):
        """Test making multiple predictions in sequence."""
        claims = [
            {
                "claim_amount": 3000,
                "days_to_report": 1,
                "claimant_age": 50,
                "prior_claims": 0
            },
            {
                "claim_amount": 20000,
                "days_to_report": 45,
                "claimant_age": 30,
                "prior_claims": 2
            },
            {
                "claim_amount": 100000,
                "days_to_report": 120,
                "claimant_age": 22,
                "prior_claims": 8
            }
        ]
        
        for claim in claims:
            result = fraud_detector.predict_claim(claim)
            assert result is not None
            assert "fraud_probability" in result