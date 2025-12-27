"""FastAPI application for insurance ML models with logging and monitoring."""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import sys
from pathlib import Path
import time

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.fraud_detector import FraudDetector
from src.logging_config import setup_logging, get_logger
from src.monitoring import monitor

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Insurance IQ API",
    description="ML-powered insurance claims analysis with monitoring",
    version="1.0.0"
)

# Load models on startup
fraud_detector = FraudDetector()

@app.on_event("startup")
async def load_models():
    """Load all models when API starts."""
    logger.info("🚀 Starting Insurance IQ API...")
    logger.info("📦 Loading models...")
    
    try:
        fraud_detector.load()
        logger.info("✅ All models loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise


# Pydantic models for request/response
class ClaimRequest(BaseModel):
    """Request model for fraud detection."""
    claim_amount: float = Field(..., description="Claim amount in dollars", gt=0)
    days_to_report: int = Field(..., description="Days between incident and report", ge=0)
    claimant_age: int = Field(..., description="Age of claimant", ge=18, le=100)
    prior_claims: int = Field(..., description="Number of prior claims", ge=0)
    
    class Config:
        json_schema_extra = {
            "example": {
                "claim_amount": 15000,
                "days_to_report": 30,
                "claimant_age": 35,
                "prior_claims": 1
            }
        }


class FraudResponse(BaseModel):
    """Response model for fraud detection."""
    prediction: str
    fraud_probability: float
    confidence: str
    risk_level: str
    recommended_action: str
    claim_data: Dict[str, Any]
    prediction_time_ms: float


# API endpoints
@app.get("/")
async def root():
    """Root endpoint - API health check."""
    logger.info("Root endpoint accessed")
    return {
        "status": "online",
        "message": "Insurance IQ API is running",
        "version": "1.0.0",
        "endpoints": {
            "fraud_detection": "/predict/fraud",
            "metrics": "/metrics",
            "health": "/health",
            "docs": "/docs"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    return {
        "status": "healthy",
        "fraud_detector_loaded": fraud_detector.is_loaded,
        "uptime_seconds": monitor.get_metrics()["uptime_seconds"]
    }


@app.get("/metrics")
async def get_metrics():
    """Get monitoring metrics.
    
    Returns:
        Current API and model metrics
    """
    logger.info("Metrics requested")
    metrics = monitor.get_metrics()
    
    # Save metrics to file
    monitor.save_metrics()
    
    return metrics


@app.post("/predict/fraud", response_model=FraudResponse)
async def predict_fraud(claim: ClaimRequest):
    """Predict fraud probability for an insurance claim.
    
    Args:
        claim: Claim data including amount, days to report, age, prior claims
        
    Returns:
        Fraud prediction with probability and recommended action
    """
    start_time = time.time()
    
    try:
        # Log incoming request
        logger.info(f"Fraud prediction requested for claim: ${claim.claim_amount}")
        
        # Convert request to dict
        claim_data = claim.model_dump()
        
        # Get prediction
        result = fraud_detector.predict_claim(claim_data)
        
        # Calculate prediction time
        prediction_time = time.time() - start_time
        result["prediction_time_ms"] = round(prediction_time * 1000, 2)
        
        # Record metrics
        monitor.record_prediction("fraud_detector", prediction_time, result)
        
        # Log result
        logger.info(
            f"Prediction: {result['prediction']} | "
            f"Probability: {result['fraud_probability']:.2%} | "
            f"Risk: {result['risk_level']} | "
            f"Time: {result['prediction_time_ms']}ms"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset monitoring metrics.
    
    Returns:
        Confirmation message
    """
    logger.warning("Metrics reset requested")
    monitor.reset_metrics()
    return {"status": "success", "message": "Metrics reset successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)