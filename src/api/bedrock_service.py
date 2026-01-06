import boto3
import json
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class BedrockService:
    """Service for interacting with Amazon Bedrock (Claude)"""
    
    def __init__(self, region_name: str = "us-east-2"):
        self.bedrock = boto3.client(
            service_name='bedrock-runtime',
            region_name=region_name
        )
        self.model_id = 'us.anthropic.claude-3-5-sonnet-20241022-v2:0'
        logger.info("BedrockService initialized")
    
    def analyze_claim(self, claim_description: str) -> Dict[str, str]:
        """
        Analyze an insurance claim using Claude via Bedrock
        
        Args:
            claim_description: Text description of the insurance claim
            
        Returns:
            Dictionary with analysis results
        """
        try:
            # Prepare the prompt
            prompt = f"""Analyze this insurance claim for potential fraud indicators:

Claim Description:
{claim_description}

Provide:
1. Risk Level (LOW/MEDIUM/HIGH)
2. Key fraud indicators (if any)
3. Recommended actions
4. Confidence level

Keep the response concise and actionable."""

            # Prepare request payload
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 1000,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
            
            # Call Bedrock
            logger.info(f"Calling Bedrock for claim analysis")
            response = self.bedrock.invoke_model(
                modelId=self.model_id,
                body=json.dumps(payload)
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            analysis = response_body['content'][0]['text']
            
            logger.info("Bedrock analysis completed successfully")
            
            return {
                "status": "success",
                "analysis": analysis,
                "model": "claude-3.5-sonnet",
                "provider": "amazon-bedrock"
            }
            
        except Exception as e:
            logger.error(f"Bedrock analysis failed: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "analysis": "Analysis unavailable"
            }