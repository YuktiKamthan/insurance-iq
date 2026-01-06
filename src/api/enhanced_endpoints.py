"""
Enhanced endpoints for Insurance IQ API
- Batch predictions
- MLflow tracking
- Demo data generation
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import time
import mlflow
import mlflow.sklearn
from datetime import datetime
import random

router = APIRouter()

# Pydantic models
class BatchClaimRequest(BaseModel):
    """Request model for batch fraud detection."""
    claims: List[Dict[str, Any]] = Field(..., description="List of claims to analyze")
    
    class Config:
        json_schema_extra = {
            "example": {
                "claims": [
                    {
                        "claim_amount": 15000,
                        "days_to_report": 30,
                        "claimant_age": 35,
                        "prior_claims": 1
                    },
                    {
                        "claim_amount": 45000,
                        "days_to_report": 90,
                        "claimant_age": 25,
                        "prior_claims": 4
                    }
                ]
            }
        }


class DemoGenerateRequest(BaseModel):
    """Request to generate demo claims."""
    num_claims: int = Field(default=10, ge=1, le=100, description="Number of claims to generate")


def create_endpoints(fraud_detector, monitor, logger):
    """
    Create enhanced endpoints with dependencies injected.
    
    Args:
        fraud_detector: Loaded fraud detection model
        monitor: Monitoring system
        logger: Logger instance
    """
    
    @router.post("/predict/fraud/batch")
    async def predict_fraud_batch(batch_request: BatchClaimRequest):
        """
        Predict fraud for multiple claims in batch.
        
        Perfect for:
        - End-of-day processing
        - Testing multiple scenarios
        - Demo presentations
        """
        start_time = time.time()
        
        try:
            logger.info(f"Batch prediction for {len(batch_request.claims)} claims")
            
            results = []
            fraud_count = 0
            high_risk_count = 0
            total_fraud_amount = 0.0
            
            for i, claim in enumerate(batch_request.claims):
                # Validate claim has required fields
                required_fields = ['claim_amount', 'days_to_report', 'claimant_age', 'prior_claims']
                if not all(field in claim for field in required_fields):
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Claim {i+1} missing required fields: {required_fields}"
                    )
                
                # Get prediction
                result = fraud_detector.predict_claim(claim)
                result["claim_index"] = i + 1
                results.append(result)
                
                # Track statistics
                if result["prediction"] == "fraud":
                    fraud_count += 1
                    total_fraud_amount += claim['claim_amount']
                
                if result["risk_level"] == "HIGH":
                    high_risk_count += 1
                
                # Record metrics
                monitor.record_prediction("fraud_detector", 0, result)
            
            # Calculate batch statistics
            batch_time = time.time() - start_time
            fraud_rate = (fraud_count / len(batch_request.claims)) * 100
            
            summary = {
                "total_claims": len(batch_request.claims),
                "fraud_detected": fraud_count,
                "fraud_rate_percent": round(fraud_rate, 2),
                "high_risk_claims": high_risk_count,
                "total_potential_fraud_amount": round(total_fraud_amount, 2),
                "batch_processing_time_ms": round(batch_time * 1000, 2),
                "avg_time_per_claim_ms": round((batch_time / len(batch_request.claims)) * 1000, 2)
            }
            
            logger.info(
                f"Batch complete: {fraud_count}/{len(batch_request.claims)} fraud detected "
                f"({fraud_rate:.1f}%) in {batch_time*1000:.0f}ms"
            )
            
            return {
                "summary": summary,
                "predictions": results
            }
            
        except Exception as e:
            logger.error(f"Batch prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")
    
    
    @router.post("/predict/fraud/tracked")
    async def predict_fraud_with_mlflow(claim: Dict[str, Any]):
        """
        Predict fraud and log to MLflow for experiment tracking.
        """
         # Set MLflow tracking URI to use the server
        mlflow.set_tracking_uri("http://mlflow:5000")
        
        start_time = time.time()
        
        try:
            # Validate claim
            required_fields = ['claim_amount', 'days_to_report', 'claimant_age', 'prior_claims']
            if not all(field in claim for field in required_fields):
                raise HTTPException(
                    status_code=400,
                    detail=f"Missing required fields: {required_fields}"
                )
            
            # Start MLflow run
            with mlflow.start_run(run_name=f"fraud_prediction_{int(time.time())}"):
                # Log input parameters
                mlflow.log_params({
                    "claim_amount": claim['claim_amount'],
                    "days_to_report": claim['days_to_report'],
                    "claimant_age": claim['claimant_age'],
                    "prior_claims": claim['prior_claims']
                })
                
                # Get prediction
                result = fraud_detector.predict_claim(claim)
                prediction_time = (time.time() - start_time) * 1000
                
                # Log prediction metrics
                mlflow.log_metrics({
                    "fraud_probability": result["fraud_probability"],
                    "prediction_time_ms": prediction_time
                })
                
                # Log prediction result
                mlflow.log_param("prediction", result["prediction"])
                mlflow.log_param("risk_level", result["risk_level"])
                mlflow.log_param("recommended_action", result["recommended_action"])
                
                # Get run ID
                run_id = mlflow.active_run().info.run_id
                
                # Add to result
                result["mlflow_run_id"] = run_id
                result["prediction_time_ms"] = round(prediction_time, 2)
                
                # Record to monitoring
                monitor.record_prediction("fraud_detector", prediction_time/1000, result)
                
                logger.info(f"Prediction logged to MLflow run: {run_id}")
                
                return result
                
        except Exception as e:
            logger.error(f"MLflow tracked prediction failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @router.post("/demo/generate-claims")
    async def generate_demo_claims(request: DemoGenerateRequest):
        """
        Generate realistic demo insurance claims for testing.
        
        Creates a mix of:
        - Low-risk legitimate claims (60%)
        - Medium-risk claims (30%)
        - High-risk suspicious claims (10%)
        """
        try:
            logger.info(f"Generating {request.num_claims} demo claims")
            
            claims = []
            
            for i in range(request.num_claims):
                # Decide risk level
                risk_roll = random.random()
                
                if risk_roll < 0.6:  # 60% low-risk
                    claim = {
                        "claim_amount": random.randint(1000, 15000),
                        "days_to_report": random.randint(1, 10),
                        "claimant_age": random.randint(35, 65),
                        "prior_claims": random.choice([0, 1])
                    }
                elif risk_roll < 0.9:  # 30% medium-risk
                    claim = {
                        "claim_amount": random.randint(15000, 35000),
                        "days_to_report": random.randint(15, 45),
                        "claimant_age": random.randint(25, 45),
                        "prior_claims": random.choice([1, 2])
                    }
                else:  # 10% high-risk
                    claim = {
                        "claim_amount": random.randint(40000, 80000),
                        "days_to_report": random.randint(60, 120),
                        "claimant_age": random.randint(20, 30),
                        "prior_claims": random.randint(3, 6)
                    }
                
                claims.append(claim)
            
            return {
                "num_claims": len(claims),
                "claims": claims,
                "distribution": {
                    "expected_low_risk": "60%",
                    "expected_medium_risk": "30%",
                    "expected_high_risk": "10%"
                }
            }
            
        except Exception as e:
            logger.error(f"Demo generation failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    
    @router.get("/demo/run-test")
    async def run_demo_test():
        """
        Run a complete demo test with generated claims.
        
        Generates 10 claims and processes them, showing:
        - Fraud detection results
        - Performance metrics
        - Summary statistics
        """
        try:
            logger.info("Running demo test")
            
            # Generate 10 demo claims
            demo_data = await generate_demo_claims(DemoGenerateRequest(num_claims=10))
            
            # Process them as a batch
            batch_result = await predict_fraud_batch(
                BatchClaimRequest(claims=demo_data["claims"])
            )
            
            return {
                "test_timestamp": datetime.now().isoformat(),
                "demo_data": demo_data,
                "results": batch_result,
                "message": "Demo test completed successfully! ✅"
            }
            
        except Exception as e:
            logger.error(f"Demo test failed: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
    
    return router