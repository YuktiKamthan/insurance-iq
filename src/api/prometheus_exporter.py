"""Simple Prometheus metrics exporter for Grafana."""

def convert_metrics_to_prometheus(metrics):
    """
    Convert JSON metrics to Prometheus format.
    
    Takes your metrics like:
      {"total_predictions": 2, "fraud": 0}
    
    Converts to Grafana format:
      total_predictions 2
      fraud_predictions 0
    """
    lines = []
    
    # Total predictions
    lines.append(f"total_predictions {metrics['total_predictions']}")
    
    # Fraud vs legitimate
    lines.append(f"fraud_predictions{{type=\"legitimate\"}} {metrics['fraud_predictions']['legitimate']}")
    lines.append(f"fraud_predictions{{type=\"fraud\"}} {metrics['fraud_predictions']['fraud']}")
    
    # Risk levels
    for level, count in metrics['risk_level_distribution'].items():
        lines.append(f"risk_level{{level=\"{level}\"}} {count}")
    
    # Performance metrics
    lines.append(f"avg_prediction_time_ms {metrics['average_prediction_time_ms']}")
    lines.append(f"uptime_seconds {metrics['uptime_seconds']}")
    
    return "\n".join(lines)
