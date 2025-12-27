"""Test the fraud detector model."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.fraud_detector import FraudDetector


def test_fraud_detector():
    """Test fraud detection on sample claims."""
    
    print("🧪 Testing Fraud Detector\n")
    print("=" * 50)
    
    # Initialize and load model
    print("\n1️⃣ Loading model...")
    detector = FraudDetector()
    detector.load()
    
    # Test cases
    test_claims = [
        {
            "name": "Suspicious Claim",
            "claim_amount": 45000,
            "days_to_report": 75,
            "claimant_age": 28,
            "prior_claims": 3
        },
        {
            "name": "Normal Claim",
            "claim_amount": 5000,
            "days_to_report": 2,
            "claimant_age": 45,
            "prior_claims": 0
        },
    ]
    
    # Test each claim
    print("\n2️⃣ Testing claims...\n")
    
    for i, claim in enumerate(test_claims, 1):
        name = claim.pop("name")
        print(f"\n{'='*50}")
        print(f"Test {i}: {name}")
        print(f"{'='*50}")
        
        result = detector.predict_claim(claim)
        
        print(f"\n📋 Claim Details:")
        print(f"   Amount: ${result['claim_data']['claim_amount']:,}")
        print(f"   Days to report: {result['claim_data']['days_to_report']}")
        
        print(f"\n🤖 Prediction:")
        print(f"   Fraud Probability: {result['fraud_probability']:.1%}")
        print(f"   Risk Level: {result['risk_level']}")
        print(f"   Action: {result['recommended_action']}")
    
    print(f"\n{'='*50}")
    print("✅ All tests completed!")


if __name__ == "__main__":
    test_fraud_detector()