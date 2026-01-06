#!/bin/bash

echo "🚀 Starting load test on insurance-iq API..."
echo "📊 Watch pods scale up with: kubectl get pods -w"
echo ""

# Port forward in background
kubectl port-forward service/insurance-iq-service 8000:8000 &
PF_PID=$!

sleep 3

echo "💥 Sending 1000 requests in parallel..."

# Send 1000 requests
for i in {1..1000}
do
  curl -s -X POST "http://localhost:8000/api/v1/analyze/claim" \
    -H "Content-Type: application/json" \
    -d '{"claim_description": "Load test claim"}' > /dev/null &
done

echo "⏳ Requests sent! Check HPA with: kubectl get hpa -w"
echo "   CPU should spike and trigger auto-scaling!"

# Wait for background jobs
wait

# Kill port-forward
kill $PF_PID 2>/dev/null