"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.api.main import app

# Create test client and trigger startup
from src.api.main import fraud_detector

# Load model for testing
fraud_detector.load()

# Create test client
client = TestClient(app)


class TestAPIEndpoints:
    """Test cases for API endpoints."""
    
    def test_root_endpoint(self):
        """Test root endpoint returns correct response."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "online"
        assert "endpoints" in data
    
    def test_health_check(self):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "fraud_detector_loaded" in data
    
    def test_metrics_endpoint(self):
        """Test metrics endpoint returns correct structure."""
        response = client.get("/metrics")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check required metric fields
        required_fields = [
            "uptime_seconds",
            "total_predictions",
            "predictions_by_model",
            "average_prediction_time_ms",
            "fraud_predictions",
            "risk_level_distribution"
        ]
        
        for field in required_fields:
            assert field in data
    
    def test_fraud_prediction_valid_claim(self):
        """Test fraud prediction with valid claim data."""
        claim = {
            "claim_amount": 15000,
            "days_to_report": 30,
            "claimant_age": 35,
            "prior_claims": 1
        }
        
        response = client.post("/predict/fraud", json=claim)
        
        assert response.status_code == 200
        data = response.json()
        
        # Check response structure
        assert "prediction" in data
        assert "fraud_probability" in data
        assert "confidence" in data
        assert "risk_level" in data
        assert "recommended_action" in data
        assert "prediction_time_ms" in data
        
        # Check value types
        assert isinstance(data["fraud_probability"], float)
        assert 0 <= data["fraud_probability"] <= 1
        assert isinstance(data["prediction_time_ms"], float)
    
    def test_fraud_prediction_invalid_amount(self):
        """Test fraud prediction with invalid claim amount."""
        claim = {
            "claim_amount": -5000,  # Invalid: negative
            "days_to_report": 30,
            "claimant_age": 35,
            "prior_claims": 1
        }
        
        response = client.post("/predict/fraud", json=claim)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_fraud_prediction_invalid_age(self):
        """Test fraud prediction with invalid age."""
        claim = {
            "claim_amount": 15000,
            "days_to_report": 30,
            "claimant_age": 10,  # Invalid: under 18
            "prior_claims": 1
        }
        
        response = client.post("/predict/fraud", json=claim)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_fraud_prediction_missing_field(self):
        """Test fraud prediction with missing required field."""
        claim = {
            "claim_amount": 15000,
            "days_to_report": 30,
            # Missing claimant_age
            "prior_claims": 1
        }
        
        response = client.post("/predict/fraud", json=claim)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_low_risk_claim(self):
        """Test that low-risk claim returns appropriate response."""
        claim = {
            "claim_amount": 3000,
            "days_to_report": 1,
            "claimant_age": 55,
            "prior_claims": 0
        }
        
        response = client.post("/predict/fraud", json=claim)
        
        assert response.status_code == 200
        data = response.json()
        assert data["risk_level"] == "LOW"
        assert data["recommended_action"] == "NORMAL_PROCESSING"
    
    def test_metrics_increment_after_prediction(self):
        """Test that metrics are updated after making predictions."""
        # Get initial metrics
        initial_response = client.get("/metrics")
        initial_data = initial_response.json()
        initial_count = initial_data["total_predictions"]
        
        # Make a prediction
        claim = {
            "claim_amount": 10000,
            "days_to_report": 15,
            "claimant_age": 40,
            "prior_claims": 1
        }
        client.post("/predict/fraud", json=claim)
        
        # Get updated metrics
        updated_response = client.get("/metrics")
        updated_data = updated_response.json()
        updated_count = updated_data["total_predictions"]
        
        # Count should have increased
        assert updated_count == initial_count + 1