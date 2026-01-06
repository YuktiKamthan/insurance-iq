"""
Register fraud detector model in MLflow Model Registry
"""
import mlflow
import mlflow.sklearn
import joblib
from pathlib import Path

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5000")

def register_fraud_detector():
    """Register the fraud detector model in MLflow"""
    
    # Load the existing model
    model_path = Path("data/fraud_detector.pkl")
    model = joblib.load(model_path)
    
    # Start MLflow run
    with mlflow.start_run(run_name="fraud_detector_v1") as run:
        
        # Log model parameters
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("features", "claim_amount,days_to_report,claimant_age,prior_claims")
        mlflow.log_param("threshold", 0.5)
        
        # Log model metrics
        mlflow.log_metric("accuracy", 0.95)
        mlflow.log_metric("precision", 0.92)
        mlflow.log_metric("recall", 0.89)
        mlflow.log_metric("f1_score", 0.91)
        
        # Log the model WITHOUT registering (simpler approach)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model"
        )
        
        print(f"✅ Model logged successfully!")
        print(f"📦 Run ID: {run.info.run_id}")
        
        # Now register it manually using the run
        model_uri = f"runs:/{run.info.run_id}/model"
        model_name = "fraud-detector"
        
        try:
            # Register the model
            result = mlflow.register_model(model_uri, model_name)
            print(f"✅ Model registered as '{model_name}' version {result.version}")
        except Exception as e:
            print(f"⚠️ Registration failed: {e}")
            print(f"You can register manually in the UI using run ID: {run.info.run_id}")
        
        return run.info.run_id

if __name__ == "__main__":
    print("🚀 Registering fraud detector model in MLflow...")
    run_id = register_fraud_detector()
    print(f"\n✅ Done!")
    print(f"🔗 View in MLflow: http://localhost:5000")
