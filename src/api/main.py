"""FastAPI application for insurance ML models with logging and monitoring."""
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Dict, Any
import sys
from pathlib import Path
import time
from bedrock_service import BedrockService
from datetime import datetime
from dotenv import load_dotenv
from snowflake_service import SnowflakeService
import uuid
# Import for Grafana metrics
from prometheus_exporter import convert_metrics_to_prometheus
from fastapi import Response
from enhanced_endpoints import create_endpoints
from drift_detector import DriftDetector

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.fraud_detector import FraudDetector
from src.logging_config import setup_logging, get_logger
from src.monitoring import monitor

# Setup logging
setup_logging(log_level="INFO")
logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Insurance IQ API",
    description="ML-powered insurance claims analysis with monitoring and explainability",
    version="1.0.0"
)

# Initialize Bedrock service
bedrock_service = BedrockService()

# Initialize Snowflake service
snowflake_service = SnowflakeService()

# Load models on startup
fraud_detector = FraudDetector()

drift_detector = DriftDetector()
logger.info("✅ Drift detector initialized successfully!")

# CREATE ENHANCED ENDPOINTS ROUTER - ADD THIS!
enhanced_router = create_endpoints(fraud_detector, monitor, logger)
app.include_router(enhanced_router, tags=["Enhanced Features"])

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
            "fraud_explanation": "/predict/fraud/explain",
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


@app.post("/predict/fraud/explain")
async def explain_fraud_prediction(claim: ClaimRequest):
    """Predict fraud probability with SHAP explanation.
    
    Args:
        claim: Claim data including amount, days to report, age, prior claims
        
    Returns:
        Fraud prediction with detailed SHAP explanation showing which features
        contributed most to the prediction
    """
    start_time = time.time()
    
    try:
        # Log incoming request
        logger.info(f"Fraud explanation requested for claim: ${claim.claim_amount}")
        
        # Convert request to dict
        claim_data = claim.model_dump()
        
        # Get prediction with explanation
        result = fraud_detector.explain_prediction(claim_data)
        
        # Calculate prediction time
        prediction_time = time.time() - start_time
        result["prediction_time_ms"] = round(prediction_time * 1000, 2)
        
        # Record metrics
        monitor.record_prediction("fraud_detector", prediction_time, result)
        
        # Log result
        logger.info(
            f"Prediction with explanation: {result['prediction']} | "
            f"Probability: {result['fraud_probability']:.2%} | "
            f"Time: {result['prediction_time_ms']}ms"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Explanation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.post("/metrics/reset")
async def reset_metrics():
    """Reset monitoring metrics.
    
    Returns:
        Confirmation message
    """
    logger.warning("Metrics reset requested")
    monitor.reset_metrics()
    return {"status": "success", "message": "Metrics reset successfully"}

@app.post("/api/v1/analyze/claim")
async def analyze_claim(request: dict):
    """
    Analyze insurance claim using Claude via Amazon Bedrock
    AND log results to Snowflake data warehouse
    """
    try:
        claim_description = request.get("claim_description", "")
        
        if not claim_description:
            raise HTTPException(
                status_code=400,
                detail="claim_description is required"
            )
        
        claim_id = str(uuid.uuid4())
        
        # Call Bedrock service
        result = bedrock_service.analyze_claim(claim_description)
        
        # Log to Snowflake
        snowflake_logged = False
        if result.get("status") == "success":
            analysis_data = {
                "claim_id": claim_id,
                "claim_description": claim_description,
                "analysis": result.get("analysis"),
                "model": result.get("model")
            }
            
            snowflake_logged = snowflake_service.insert_bedrock_analysis(analysis_data)
            
            if snowflake_logged:
                logger.info(f"Successfully logged analysis to Snowflake for claim {claim_id}")
        
        return {
            "claim_id": claim_id,
            "claim_description": claim_description,
            "bedrock_analysis": result,
            "snowflake_logged": snowflake_logged,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Claim analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))    



@app.get("/api/v1/snowflake/recent-claims")
async def get_recent_claims(limit: int = 10):
    """
    Get recent claims from Snowflake data warehouse
    """
    try:
        claims = snowflake_service.get_recent_claims(limit=limit)
        return {
            "claims": claims,
            "count": len(claims),
            "source": "snowflake"
        }
    except Exception as e:
        logger.error(f"Failed to get recent claims: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/snowflake/high-risk-claims")
async def get_high_risk_claims(limit: int = 10):
    """
    Get high-risk claims from Snowflake based on AI analysis
    """
    try:
        claims = snowflake_service.get_high_risk_claims(limit=limit)
        return {
            "high_risk_claims": claims,
            "count": len(claims),
            "source": "snowflake"
        }
    except Exception as e:
        logger.error(f"Failed to get high-risk claims: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/snowflake/submit-claim")
async def submit_claim(request: dict):
    """
    Submit a new claim directly to Snowflake
    """
    try:
        claim_data = {
            "claim_id": str(uuid.uuid4()),
            "claim_description": request.get("claim_description"),
            "claim_amount": request.get("claim_amount"),
            "customer_id": request.get("customer_id", "CUST_" + str(uuid.uuid4())[:8]),
            "claim_date": datetime.now()
        }
        
        success = snowflake_service.insert_claim(claim_data)
        
        if success:
            return {
                "status": "success",
                "claim_id": claim_data["claim_id"],
                "message": "Claim submitted to Snowflake successfully"
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to submit claim to Snowflake")
            
    except Exception as e:
        logger.error(f"Failed to submit claim: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))                

@app.get("/metrics/prometheus")
async def prometheus_metrics():
    """
    Prometheus metrics endpoint for Grafana.
    
    Returns metrics in Prometheus format that Grafana can scrape.
    """
    # Get existing metrics in JSON format
    metrics = monitor.get_metrics()
    
    # Convert to Prometheus format
    prometheus_format = convert_metrics_to_prometheus(metrics)
    
    # Return as plain text (what Grafana expects)
    return Response(content=prometheus_format, media_type="text/plain")

# ================== DRIFT DETECTION ==================
@app.post("/drift/check")
async def check_drift(claims: List[Dict[str, Any]]):
    """
    Check for data drift in recent claims.
    
    Compares recent claims against training data baseline to detect
    distribution shifts that might affect model performance.
    """
    try:
        logger.info(f"🔍 Checking drift for {len(claims)} claims")
        
        # Extract claim data (features only)
        claim_features = []
        for claim in claims:
            claim_features.append({
                'claim_amount': claim.get('claim_amount'),
                'days_to_report': claim.get('days_to_report'),
                'claimant_age': claim.get('claimant_age'),
                'prior_claims': claim.get('prior_claims')
            })
        
        # Run drift detection
        drift_report = drift_detector.check_drift(claim_features)
        
        logger.info(f"✅ Drift check complete: {drift_report['status']}")
        
        return drift_report
        
    except Exception as e:
        logger.error(f"❌ Drift detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Drift detection error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
