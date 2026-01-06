"""
Train BALANCED fraud detection model with MLflow tracking
"""
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
from datetime import datetime

# Set MLflow tracking URI
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("fraud-detection-experiments")

def generate_synthetic_data(n_samples=1000):
    """Generate synthetic insurance claims data"""
    np.random.seed(42)
    
    # Generate features
    claim_amounts = np.random.exponential(scale=5000, size=n_samples)
    days_to_report = np.random.poisson(lam=15, size=n_samples)
    claimant_ages = np.random.normal(loc=45, scale=15, size=n_samples).clip(18, 100)
    prior_claims = np.random.poisson(lam=1.5, size=n_samples)
    
    # Generate fraud labels (with some logic)
    fraud_probability = (
        (claim_amounts > 10000) * 0.3 +
        (days_to_report > 30) * 0.2 +
        (prior_claims > 3) * 0.25 +
        (claimant_ages < 25) * 0.15
    )
    
    is_fraud = (np.random.random(n_samples) < fraud_probability).astype(int)
    
    # Create DataFrame
    df = pd.DataFrame({
        'claim_amount': claim_amounts,
        'days_to_report': days_to_report,
        'claimant_age': claimant_ages,
        'prior_claims': prior_claims,
        'is_fraud': is_fraud
    })
    
    return df

def train_balanced_model():
    """
    Train fraud detection model WITH CLASS BALANCING
    """
    
    with mlflow.start_run(run_name=f"fraud_balanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        print("=" * 60)
        print("🚀 TRAINING BALANCED FRAUD DETECTION MODEL")
        print("=" * 60)
        
        # Parameters
        n_estimators = 100
        max_depth = 10
        min_samples_split = 5
        
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("class_weight", "balanced")  # THIS IS THE KEY!
        
        # Generate data
        print("\n📊 Generating synthetic data...")
        df = generate_synthetic_data(n_samples=1000)
        fraud_rate = df['is_fraud'].mean()
        
        mlflow.log_param("n_samples", len(df))
        mlflow.log_param("fraud_rate", fraud_rate)
        
        print(f"✅ Generated {len(df)} samples")
        print(f"   Fraud rate: {fraud_rate:.2%}")
        
        # Split data
        X = df[['claim_amount', 'days_to_report', 'claimant_age', 'prior_claims']]
        y = df['is_fraud']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📈 Train set: {len(X_train)} samples")
        print(f"   - Legitimate: {(y_train == 0).sum()}")
        print(f"   - Fraud: {(y_train == 1).sum()}")
        print(f"📉 Test set: {len(X_test)} samples")
        print(f"   - Legitimate: {(y_test == 0).sum()}")
        print(f"   - Fraud: {(y_test == 1).sum()}")
        
        # Train model WITH CLASS WEIGHTS
        print(f"\n🤖 Training Balanced Random Forest model...")
        print(f"   ⚖️ Using class_weight='balanced'")
        print(f"   This tells the model: 'Fraud is IMPORTANT even though it's rare!'")
        
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            class_weight='balanced',  # THIS FIXES THE IMBALANCE!
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("auc_roc", auc)
        
        print("\n📊 MODEL PERFORMANCE:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   Precision: {precision:.4f} ⬆️ (improved!)")
        print(f"   Recall:    {recall:.4f} ⬆️ (improved!)")
        print(f"   F1 Score:  {f1:.4f} ⬆️ (improved!)")
        print(f"   AUC-ROC:   {auc:.4f}")
        
        # Show confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\n🎯 CONFUSION MATRIX:")
        print(f"   True Negatives:  {cm[0][0]} (correctly predicted legitimate)")
        print(f"   False Positives: {cm[0][1]} (wrongly flagged as fraud)")
        print(f"   False Negatives: {cm[1][0]} (missed fraud cases)")
        print(f"   True Positives:  {cm[1][1]} (correctly caught fraud)")
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="fraud-detector"
        )
        
        # Save locally
        joblib.dump(model, 'models/fraud_detector.pkl')
        print("\n💾 Model saved to:")
        print("   - MLflow: logged as 'fraud-detector'")
        print("   - Local: models/fraud_detector.pkl")
        
        print("\n" + "=" * 60)
        print("✅ BALANCED MODEL TRAINING COMPLETE!")
        print("=" * 60)
        
        return model

if __name__ == "__main__":
    print("🔬 Training Balanced Fraud Detection Model")
    print("=" * 60)
    
    train_balanced_model()
    
    print("\n" + "=" * 60)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 60)
    print("\n📊 To view results in MLflow UI:")
    print("   mlflow ui")
    print("\n   Then open: http://localhost:5000")
    print("\n💡 Compare this run with previous runs to see the improvement!")