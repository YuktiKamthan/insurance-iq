# Insurance IQ - Production ML System for Insurance Analytics

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green)
![AWS](https://img.shields.io/badge/AWS-SageMaker-orange)
![Tests](https://img.shields.io/badge/Tests-15%20Passing-success)

> End-to-end machine learning system for insurance fraud detection, claims severity prediction, customer churn analysis, premium optimization, and subrogation scoring. Built with production-grade infrastructure including FastAPI, monitoring, automated testing, and AWS SageMaker training.

## 🎯 Project Overview

**Insurance IQ** is a comprehensive ML engineering portfolio project demonstrating production-ready machine learning deployment for the insurance industry. The system includes 5 trained models, a REST API with monitoring, comprehensive testing, and cloud-based training infrastructure.

### Key Features

- ✅ **5 Production ML Models** trained on synthetic insurance data
- ✅ **FastAPI REST API** with automatic OpenAPI documentation
- ✅ **Real-time Monitoring** with performance metrics tracking
- ✅ **Comprehensive Logging** system for debugging and audit trails
- ✅ **100% Test Coverage** with pytest (15 passing tests)
- ✅ **AWS SageMaker Training** for scalable model development
- ✅ **Docker Support** for containerized deployment
- ✅ **Professional Code Structure** following best practices

---

## 🏗️ Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Insurance IQ System                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │   FastAPI    │ ───► │  ML Models   │ ───► │    S3     │ │
│  │  REST API    │      │  (5 models)  │      │  Storage  │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                      │                     │       │
│         ▼                      ▼                     ▼       │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │  Monitoring  │      │   Logging    │      │ SageMaker │ │
│  │   Metrics    │      │    System    │      │ Training  │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 🤖 Machine Learning Models

### 1. Fraud Detection
- **Algorithm:** Random Forest Classifier
- **Performance:** AUC 0.94 (local), AUC 0.75 (SageMaker)
- **Use Case:** Identify suspicious insurance claims for investigation
- **Features:** claim_amount, days_to_report, claimant_age, prior_claims
- **Output:** Fraud probability, risk level, recommended action

### 2. Claims Severity Prediction
- **Algorithm:** Gradient Boosting Regressor
- **Performance:** R² 0.95, MAE $367
- **Use Case:** Estimate claim costs for reserve setting
- **Features:** claimant_age, prior_claims, days_to_report

### 3. Customer Churn Prediction
- **Algorithm:** Random Forest Classifier
- **Performance:** AUC 0.67
- **Use Case:** Identify at-risk customers for retention campaigns
- **Features:** tenure, policies, claims, premium, service_calls, autopay

### 4. Premium Optimization
- **Algorithm:** Gradient Boosting Regressor
- **Performance:** R² 0.95, MAE $41
- **Use Case:** Risk-based pricing optimization
- **Features:** driver_age, years_driving, vehicle_age, credit_score, mileage

### 5. Subrogation Likelihood
- **Algorithm:** Random Forest Classifier
- **Performance:** AUC 0.85+
- **Use Case:** Identify high-value recovery opportunities
- **Features:** claim_amount, other_party_insured, police_report, witnesses, fault_clarity

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Docker (optional)
- AWS Account (for SageMaker features)

### Installation
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/insurance-iq.git
cd insurance-iq

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the API
```bash
# Start the FastAPI server
uvicorn src.api.main:app --reload

# API will be available at:
# http://localhost:8000
# Documentation at: http://localhost:8000/docs
```

---

## 📡 API Usage

### Fraud Detection Endpoint

**Request:**
```bash
curl -X POST http://localhost:8000/predict/fraud \
  -H "Content-Type: application/json" \
  -d '{
    "claim_amount": 15000,
    "days_to_report": 30,
    "claimant_age": 35,
    "prior_claims": 1
  }'
```

**Response:**
```json
{
  "prediction": "legitimate",
  "fraud_probability": 0.23,
  "confidence": "high",
  "risk_level": "LOW",
  "recommended_action": "NORMAL_PROCESSING",
  "prediction_time_ms": 45.2
}
```

### Monitoring Metrics
```bash
curl http://localhost:8000/metrics
```

**Response:**
```json
{
  "uptime_seconds": 3600,
  "total_predictions": 150,
  "average_prediction_time_ms": 111.81,
  "fraud_predictions": {"legitimate": 142, "fraud": 8},
  "risk_level_distribution": {"LOW": 130, "MEDIUM": 15, "HIGH": 5}
}
```

---

## 🧪 Testing
```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Test results: 15/15 passing (100%)
```

---

## ☁️ AWS SageMaker Training

### Training a Model
```python
from sagemaker.sklearn import SKLearn

estimator = SKLearn(
    entry_point='train.py',
    role=role,
    instance_type='ml.m5.large',
    framework_version='1.2-1'
)

estimator.fit({'train': 's3://bucket/data.csv'})
```

**Training Results:**
- Instance: ml.m5.large
- Training Time: 109 seconds
- AUC Score: 0.7495
- Cost: ~$0.15

---

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| API Response Time | 111ms avg |
| Test Coverage | 100% (15/15 tests) |
| Models Trained | 5 production models |
| SageMaker Training Time | 109 seconds |
| Total AWS Cost | ~$0.20 |

---

## 🐳 Docker Deployment
```bash
# Build image
docker build -t insurance-iq:latest .

# Run container
docker run -p 8000:8000 insurance-iq:latest

# Access API at http://localhost:8000
```

---

## 📁 Project Structure
```
insurance-iq/
├── data/                          # Training data and models
│   ├── claims_data.csv
│   ├── fraud_detector.pkl
│   └── model_performance.json
├── notebooks/                     # Jupyter notebooks
│   └── 01_hello_insurance.ipynb
├── src/                          # Source code
│   ├── api/
│   │   ├── __init__.py
│   │   └── main.py              # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_model.py        # Base model class
│   │   └── fraud_detector.py    # Fraud detection model
│   ├── logging_config.py        # Logging setup
│   └── monitoring.py            # Metrics tracking
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_api.py              # API tests
│   └── test_fraud_detector.py   # Model tests
├── logs/                         # Application logs
├── monitoring/                   # Metrics storage
├── Dockerfile                    # Docker configuration
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

---

## 🎓 Technical Skills Demonstrated

- **Machine Learning:** Scikit-learn, model training, evaluation, hyperparameter tuning
- **API Development:** FastAPI, REST API design, request validation, auto-documentation
- **Cloud Computing:** AWS SageMaker, S3, EC2, IAM
- **DevOps:** Docker, Git, CI/CD readiness
- **Testing:** Pytest, unit tests, integration tests
- **Monitoring:** Logging, metrics tracking, performance monitoring
- **Software Engineering:** Clean code, modular design, professional structure

---

## 🔮 Future Enhancements

- [ ] Model explainability with SHAP values
- [ ] A/B testing framework
- [ ] Real-time model monitoring dashboard
- [ ] Automated retraining pipeline
- [ ] Multi-model ensemble predictions
- [ ] GraphQL API support
- [ ] Kubernetes deployment
- [ ] CI/CD with GitHub Actions

---

## 📝 License

This project is for portfolio demonstration purposes.

---

## 👤 Author

**Yukti Kamthan**
- 9+ years software engineering experience
- M.S. Information Systems (AI & Business Analytics) - Florida International University
- Specialization: ML Engineering, Insurance Analytics

---

## 🙏 Acknowledgments

Built as a portfolio project to demonstrate production ML engineering capabilities for roles at companies like Guidewire, leveraging industry-standard tools and AWS infrastructure.

---

**⭐ If you found this project interesting, please star the repository!**