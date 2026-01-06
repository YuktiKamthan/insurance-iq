"""
Data drift detection using Evidently AI
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any
from evidently.report import Report
from evidently.metrics import DatasetDriftMetric, ColumnDriftMetric

class DriftDetector:
    """Detect data drift in insurance claims"""
    
    def __init__(self):
        """Initialize drift detector with reference baseline"""
        self.feature_columns = ['claim_amount', 'days_to_report', 'claimant_age', 'prior_claims']
        self.reference_data = self._create_reference_baseline()
        
    def _create_reference_baseline(self) -> pd.DataFrame:
        """
        Create reference baseline data representing training distribution.
        In production, you'd load actual training data.
        """
        # Simulating training data distribution
        np.random.seed(42)
        
        n_samples = 1000
        
        data = {
            'claim_amount': np.random.gamma(2, 5000, n_samples),  # Right-skewed
            'days_to_report': np.random.exponential(20, n_samples),  # Exponential
            'claimant_age': np.random.normal(45, 15, n_samples),  # Normal distribution
            'prior_claims': np.random.poisson(1.5, n_samples)  # Poisson
        }
        
        df = pd.DataFrame(data)
        
        # Ensure positive values
        df['claim_amount'] = df['claim_amount'].clip(lower=1000)
        df['days_to_report'] = df['days_to_report'].clip(lower=0)
        df['claimant_age'] = df['claimant_age'].clip(lower=18, upper=100)
        df['prior_claims'] = df['prior_claims'].clip(lower=0)
        
        return df
    
    def check_drift(self, claims: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check for data drift in recent claims
        
        Args:
            claims: List of claim dictionaries
            
        Returns:
            Drift detection report
        """
        # Convert to DataFrame
        current_df = pd.DataFrame(claims)
        
        # Validate columns
        if not all(col in current_df.columns for col in self.feature_columns):
            return {
                "error": "Missing required columns",
                "required": self.feature_columns,
                "received": list(current_df.columns)
            }
        
        # Select only feature columns
        current_df = current_df[self.feature_columns]
        reference_df = self.reference_data[self.feature_columns]
        
        # Run Evidently drift detection
        report = Report(metrics=[
            DatasetDriftMetric(),
            ColumnDriftMetric(column_name='claim_amount'),
            ColumnDriftMetric(column_name='days_to_report'),
            ColumnDriftMetric(column_name='claimant_age'),
            ColumnDriftMetric(column_name='prior_claims')
        ])
        
        report.run(reference_data=reference_df, current_data=current_df)
        
        # Extract results
        results = report.as_dict()
        
        # Parse drift results with NaN handling
        drift_summary = self._parse_drift_results(results, current_df, reference_df)
        
        return drift_summary
    
    def _parse_drift_results(self, results: Dict, current_df: pd.DataFrame, reference_df: pd.DataFrame) -> Dict:
        """Parse Evidently results into actionable summary"""
        
        # Extract dataset-level drift
        dataset_drift = results['metrics'][0]['result'].get('dataset_drift', False)
        
        # Extract feature-level drift
        feature_drift = {}
        
        for i, feature in enumerate(self.feature_columns, start=1):
            metric = results['metrics'][i]['result']
            
            # Safely get drift score (p-value), handle NaN
            drift_score = metric.get('drift_score', 1.0)
            if np.isnan(drift_score) or np.isinf(drift_score):
                drift_score = 1.0  # Default to no drift if NaN
            
            drift_detected = metric.get('drift_detected', False)
            
            # Calculate statistics with NaN handling
            current_mean = float(current_df[feature].mean())
            reference_mean = float(reference_df[feature].mean())
            current_std = float(current_df[feature].std())
            reference_std = float(reference_df[feature].std())
            
            # Replace NaN/inf with defaults
            if np.isnan(current_mean) or np.isinf(current_mean):
                current_mean = 0.0
            if np.isnan(reference_mean) or np.isinf(reference_mean):
                reference_mean = 0.0
            if np.isnan(current_std) or np.isinf(current_std):
                current_std = 0.0
            if np.isnan(reference_std) or np.isinf(reference_std):
                reference_std = 0.0
            
            feature_drift[feature] = {
                'drift_detected': drift_detected,
                'drift_score': float(drift_score),
                'current_mean': current_mean,
                'reference_mean': reference_mean,
                'current_std': current_std,
                'reference_std': reference_std
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(feature_drift)
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'drift_detected': dataset_drift,
            'n_samples_analyzed': len(current_df),
            'n_features': len(self.feature_columns),
            'feature_drift': feature_drift,
            'recommendations': recommendations,
            'status': 'ALERT' if dataset_drift else 'OK'
        }
        
        return summary
    
    def _generate_recommendations(self, feature_drift: Dict) -> List[str]:
        """Generate actionable recommendations based on drift detection."""
        recommendations = []
        
        for feature, drift_info in feature_drift.items():
            if drift_info['drift_detected']:
                # Calculate percent change safely
                ref_mean = drift_info['reference_mean']
                curr_mean = drift_info['current_mean']
                
                if ref_mean != 0:
                    pct_change = ((curr_mean - ref_mean) / ref_mean * 100)
                else:
                    pct_change = 0.0
                
                if abs(pct_change) > 20:
                    recommendations.append(
                        f"⚠️ {feature}: Significant shift detected ({pct_change:+.1f}%). "
                        f"Consider retraining model with recent data."
                    )
        
        if not recommendations:
            recommendations.append("✅ No significant drift detected. Model performance should be stable.")
        
        return recommendations
