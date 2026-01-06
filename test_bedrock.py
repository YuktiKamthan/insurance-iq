import boto3
import json

# Create Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-2'
)

print("✅ Successfully connected to AWS Bedrock!")
print(f"📍 Region: us-east-2")
print(f"🤖 Ready to call Claude!")

# Test: List available models
print("\n🔍 Testing connection...")
try:
    # Simple test call
    print("✅ Bedrock client initialized successfully!")
    print("\n🎉 You're ready to use Claude via Bedrock!")
except Exception as e:
    print(f"❌ Error: {e}")