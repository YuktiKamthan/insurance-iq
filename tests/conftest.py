
import pytest
import sys
from pathlib import Path

# Add src directories to path BEFORE any imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root / "src" / "api"))
sys.path.insert(0, str(project_root / "src" / "models"))

# Now we can import
from fastapi.testclient import TestClient

# Import app - this will trigger model loading
from main import app

@pytest.fixture(scope="session")
def client():
    """Create a test client for the API"""
    return TestClient(app)

@pytest.fixture
def sample_claim():
    """Sample claim for testing"""
    return {
        "claim_amount": 15000,
        "days_to_report": 30,
        "claimant_age": 35,
        "prior_claims": 1
    }

@pytest.fixture
def sample_claims_batch():
    """Sample batch of claims for testing"""
    return [
        {"claim_amount": 15000, "days_to_report": 5, "claimant_age": 35, "prior_claims": 1},
        {"claim_amount": 25000, "days_to_report": 10, "claimant_age": 45, "prior_claims": 2},
        {"claim_amount": 8000, "days_to_report": 3, "claimant_age": 28, "prior_claims": 0}
    ]

@pytest.fixture
def high_fraud_claim():
    """Claim likely to be flagged as fraud"""
    return {
        "claim_amount": 75000,
        "days_to_report": 95,
        "claimant_age": 23,
        "prior_claims": 6
    }

@pytest.fixture
def low_fraud_claim():
    """Claim likely to be legitimate"""
    return {
        "claim_amount": 5000,
        "days_to_report": 2,
        "claimant_age": 45,
        "prior_claims": 0
    }
