"""Monitoring and metrics tracking for ML models."""
import time
from typing import Dict, Any
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path


class ModelMonitor:
    """Monitor model predictions and performance."""
    
    def __init__(self):
        """Initialize monitoring."""
        self.predictions_count = defaultdict(int)
        self.prediction_times = []
        self.fraud_predictions = {"legitimate": 0, "fraud": 0}
        self.risk_levels = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        self.start_time = datetime.now()
        
        # Create monitoring directory
        self.monitor_dir = Path("monitoring")
        self.monitor_dir.mkdir(exist_ok=True)
    
    def record_prediction(
        self,
        model_name: str,
        prediction_time: float,
        result: Dict[str, Any]
    ) -> None:
        """Record a prediction for monitoring.
        
        Args:
            model_name: Name of the model used
            prediction_time: Time taken to make prediction (seconds)
            result: Prediction result dictionary
        """
        # Track counts
        self.predictions_count[model_name] += 1
        self.prediction_times.append(prediction_time)
        
        # Track fraud-specific metrics
        if model_name == "fraud_detector":
            self.fraud_predictions[result["prediction"]] += 1
            self.risk_levels[result["risk_level"]] += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics.
        
        Returns:
            Dictionary with current metrics
        """
        total_predictions = sum(self.predictions_count.values())
        avg_time = (
            sum(self.prediction_times) / len(self.prediction_times)
            if self.prediction_times else 0
        )
        
        uptime = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "uptime_seconds": uptime,
            "total_predictions": total_predictions,
            "predictions_by_model": dict(self.predictions_count),
            "average_prediction_time_ms": round(avg_time * 1000, 2),
            "min_prediction_time_ms": (
                round(min(self.prediction_times) * 1000, 2)
                if self.prediction_times else 0
            ),
            "max_prediction_time_ms": (
                round(max(self.prediction_times) * 1000, 2)
                if self.prediction_times else 0
            ),
            "fraud_predictions": dict(self.fraud_predictions),
            "risk_level_distribution": dict(self.risk_levels),
            "last_updated": datetime.now().isoformat()
        }
    
    def save_metrics(self) -> None:
        """Save metrics to file."""
        metrics = self.get_metrics()
        metrics_file = self.monitor_dir / "metrics.json"
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def reset_metrics(self) -> None:
        """Reset all metrics."""
        self.predictions_count.clear()
        self.prediction_times.clear()
        self.fraud_predictions = {"legitimate": 0, "fraud": 0}
        self.risk_levels = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
        self.start_time = datetime.now()


# Global monitor instance
monitor = ModelMonitor()