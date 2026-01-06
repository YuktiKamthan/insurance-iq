# Kubernetes Deployment Guide

## Prerequisites
- Kubernetes cluster running (Docker Desktop, Minikube, or cloud)
- kubectl installed and configured
- Docker image `insurance-iq:latest` built

## Quick Deploy

### 1. Create Secrets
```bash
cd k8s
chmod +x create-secrets.sh
./create-secrets.sh
```

### 2. Deploy Application
```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
```

### 3. Verify Deployment
```bash
# Check pods are running
kubectl get pods

# Check service
kubectl get services

# View logs
kubectl logs -l app=insurance-iq

# Get service URL
kubectl get service insurance-iq-service
```

### 4. Test API
```bash
# Forward port to local machine
kubectl port-forward service/insurance-iq-service 8000:8000

# In another terminal, test
curl http://localhost:8000/health
```

## Scaling
```bash
# Scale to 5 replicas
kubectl scale deployment insurance-iq-deployment --replicas=5

# Auto-scale based on CPU
kubectl autoscale deployment insurance-iq-deployment --min=2 --max=10 --cpu-percent=80
```

## Cleanup
```bash
kubectl delete -f service.yaml
kubectl delete -f deployment.yaml
kubectl delete secret aws-credentials
kubectl delete secret insurance-iq-secrets
```

## Architecture
```
User Request → Service (LoadBalancer) → Pods (3 replicas)
                                         ├─ Pod 1
                                         ├─ Pod 2
                                         └─ Pod 3
```
```

---

**✋ Create this file!**

---

## **NOW LET'S CHECK YOUR K8S FOLDER:**

You should have these files:
```
k8s/
├── deployment.yaml
├── service.yaml
├── secrets-template.yaml
├── create-secrets.sh
└── README.md