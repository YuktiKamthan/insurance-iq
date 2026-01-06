#!/bin/bash
# Script to create Kubernetes secrets from your existing credentials

echo "🔐 Creating Kubernetes Secrets..."

# 1. Create AWS credentials secret from your ~/.aws directory
echo "📝 Creating AWS credentials secret..."
kubectl create secret generic aws-credentials \
  --from-file=credentials=$HOME/.aws/credentials \
  --from-file=config=$HOME/.aws/config \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Create Snowflake credentials secret from .env file
echo "📝 Creating Snowflake credentials secret..."

# Read from .env file
if [ -f ../.env ]; then
  source ../.env
  
  kubectl create secret generic insurance-iq-secrets \
    --from-literal=snowflake-account="$SNOWFLAKE_ACCOUNT" \
    --from-literal=snowflake-user="$SNOWFLAKE_USER" \
    --from-literal=snowflake-password="$SNOWFLAKE_PASSWORD" \
    --from-literal=snowflake-warehouse="$SNOWFLAKE_WAREHOUSE" \
    --from-literal=snowflake-database="$SNOWFLAKE_DATABASE" \
    --from-literal=snowflake-schema="$SNOWFLAKE_SCHEMA" \
    --from-literal=snowflake-role="$SNOWFLAKE_ROLE" \
    --dry-run=client -o yaml | kubectl apply -f -
  
  echo "✅ Secrets created successfully!"
else
  echo "❌ Error: .env file not found!"
  exit 1
fi

echo ""
echo "📋 Verify secrets with:"
echo "  kubectl get secrets"
echo "  kubectl describe secret aws-credentials"
echo "  kubectl describe secret insurance-iq-secrets"