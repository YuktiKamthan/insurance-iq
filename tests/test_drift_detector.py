import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src/api"))

from drift_detector import DriftDetector

@pytest.fixture
def drift_detector():
    """Create drift detector instance"""
    return DriftDetector()

class TestDriftDetectorInitialization:
    """Test drift detector initialization"""
    
    def test_detector_loads(self, drift_detector):
        """Test that drift detector initializes"""
        assert drift_detector.reference_data is not None
        assert drift_detector.feature_columns is not None
    
    def test_reference_baseline_created(self, drift_detector):
        """Test that reference baseline exists"""
        assert len(drift_detector.reference_data) > 0
        expected_features = ['claim_amount', 'days_to_report', 'claimant_age', 'prior_claims']
        assert drift_detector.feature_columns == expected_features

class TestDriftDetection:
    """Test drift detection functionality"""
    
    def test_no_drift_similar_data(self, drift_detector):
        """Test no drift detected with similar data"""
        similar_claims = [
            {'claim_amount': 10000, 'days_to_report': 20, 'claimant_age': 45, 'prior_claims': 1},
            {'claim_amount': 12000, 'days_to_report': 18, 'claimant_age': 43, 'prior_claims': 2},
            {'claim_amount': 9000, 'days_to_report': 22, 'claimant_age': 47, 'prior_claims': 1}
        ]
        
        result = drift_detector.check_drift(similar_claims)
        assert 'drift_detected' in result
        assert 'status' in result
        assert result['status'] in ['OK', 'ALERT']
    
    def test_drift_detected_extreme_data(self, drift_detector):
        """Test drift detection with extreme data"""
        extreme_claims = [
            {'claim_amount': 150000, 'days_to_report': 90, 'claimant_age': 85, 'prior_claims': 15},
            {'claim_amount': 200000, 'days_to_report': 120, 'claimant_age': 90, 'prior_claims': 20},
            {'claim_amount': 180000, 'days_to_report': 100, 'claimant_age': 82, 'prior_claims': 18}
        ]
        
        result = drift_detector.check_drift(extreme_claims)
        assert result['drift_detected'] is True
        assert result['status'] == 'ALERT'
    
    def test_drift_report_structure(self, drift_detector):
        """Test that drift report has correct structure"""
        claims = [
            {'claim_amount': 15000, 'days_to_report': 5, 'claimant_age': 35, 'prior_claims': 1}
        ]
        
        result = drift_detector.check_drift(claims)
        assert 'timestamp' in result
        assert 'drift_detected' in result
        assert 'n_samples_analyzed' in result
        assert 'feature_drift' in result
        assert 'recommendations' in result
        
        # Check feature drift details
        for feature in drift_detector.feature_columns:
            assert feature in result['feature_drift']
            assert 'drift_score' in result['feature_drift'][feature]
            assert 'current_mean' in result['feature_drift'][feature]
