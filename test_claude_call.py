import boto3
import json

# Create Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-2'
)

print("🤖 Testing Claude via Bedrock...\n")

# Prepare the request
claim_description = "Customer reported car accident 3 months after it happened. Claiming $45,000 in damages. No police report filed."

# Format the message for Claude
payload = {
    "anthropic_version": "bedrock-2023-05-31",
    "max_tokens": 1000,
    "messages": [
        {
            "role": "user",
            "content": f"Analyze this insurance claim for potential fraud indicators:\n\n{claim_description}\n\nProvide a brief risk assessment."
        }
    ]
}

print(f"📋 Claim: {claim_description}\n")
print("⏳ Calling Claude...\n")

try:
    # Call Claude 3.5 Sonnet
    response = bedrock.invoke_model(
        modelId='us.anthropic.claude-3-5-sonnet-20241022-v2:0',
        body=json.dumps(payload)
    )
    
    # Parse response
    response_body = json.loads(response['body'].read())
    claude_response = response_body['content'][0]['text']
    
    print("✅ Claude's Analysis:")
    print("=" * 60)
    print(claude_response)
    print("=" * 60)
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nIf you see an access error, we may need to enable the model.")