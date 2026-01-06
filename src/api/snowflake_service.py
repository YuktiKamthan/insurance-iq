import snowflake.connector
from typing import Dict, List, Optional
import logging
import os
from datetime import datetime
import uuid

logger = logging.getLogger(__name__)


class SnowflakeService:
    """Service for interacting with Snowflake data warehouse"""
    
    def __init__(self):
        """Initialize Snowflake connection using environment variables"""
        self.account = os.getenv('SNOWFLAKE_ACCOUNT', 'ewc54933.us-east-1')
        self.user = os.getenv('SNOWFLAKE_USER')
        self.password = os.getenv('SNOWFLAKE_PASSWORD')
        self.warehouse = os.getenv('SNOWFLAKE_WAREHOUSE', 'COMPUTE_WH')
        self.database = os.getenv('SNOWFLAKE_DATABASE', 'INSURANCE_IQ')
        self.schema = os.getenv('SNOWFLAKE_SCHEMA', 'CLAIMS_DATA')
        self.role = os.getenv('SNOWFLAKE_ROLE', 'ACCOUNTADMIN')
        
        self.connection = None
        logger.info("SnowflakeService initialized")
    
    def connect(self):
        """Establish connection to Snowflake"""
        try:
            if not self.user or not self.password:
                logger.error("Snowflake credentials not set in environment variables")
                return False
            
            self.connection = snowflake.connector.connect(
                user=self.user,
                password=self.password,
                account=self.account,
                warehouse=self.warehouse,
                database=self.database,
                schema=self.schema,
                role=self.role
            )
            logger.info("Successfully connected to Snowflake")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Snowflake: {str(e)}")
            return False
    
    def disconnect(self):
        """Close Snowflake connection"""
        if self.connection:
            self.connection.close()
            logger.info("Disconnected from Snowflake")
    
    def insert_claim(self, claim_data: Dict) -> bool:
        """
        Insert claim data into Snowflake
        
        Args:
            claim_data: Dictionary with claim information
        """
        try:
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            
            claim_id = claim_data.get('claim_id', str(uuid.uuid4()))
            
            insert_query = """
            INSERT INTO CLAIMS (
                CLAIM_ID, CLAIM_DESCRIPTION, CLAIM_AMOUNT, 
                CUSTOMER_ID, CLAIM_DATE
            ) VALUES (%s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                claim_id,
                claim_data.get('claim_description'),
                claim_data.get('claim_amount'),
                claim_data.get('customer_id'),
                claim_data.get('claim_date', datetime.now())
            ))
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Inserted claim {claim_id} into Snowflake")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert claim: {str(e)}")
            return False
    
    def insert_fraud_prediction(self, prediction_data: Dict) -> bool:
        """
        Insert fraud prediction into Snowflake
        
        Args:
            prediction_data: Dictionary with prediction results
        """
        try:
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            
            prediction_id = str(uuid.uuid4())
            
            insert_query = """
            INSERT INTO FRAUD_PREDICTIONS (
                PREDICTION_ID, CLAIM_ID, FRAUD_PROBABILITY, 
                PREDICTION, MODEL_VERSION
            ) VALUES (%s, %s, %s, %s, %s)
            """
            
            cursor.execute(insert_query, (
                prediction_id,
                prediction_data.get('claim_id'),
                prediction_data.get('fraud_probability'),
                prediction_data.get('prediction'),
                prediction_data.get('model_version', 'v1.0')
            ))
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Inserted fraud prediction {prediction_id} into Snowflake")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert fraud prediction: {str(e)}")
            return False
    
    def insert_bedrock_analysis(self, analysis_data: Dict) -> bool:
        """
        Insert Bedrock AI analysis into Snowflake
        
        Args:
            analysis_data: Dictionary with AI analysis results
        """
        try:
            if not self.connection:
                if not self.connect():
                    return False
            
            cursor = self.connection.cursor()
            
            analysis_id = str(uuid.uuid4())
            
            insert_query = """
            INSERT INTO BEDROCK_ANALYSIS (
                ANALYSIS_ID, CLAIM_ID, CLAIM_DESCRIPTION, 
                RISK_LEVEL, ANALYSIS_TEXT, MODEL_NAME
            ) VALUES (%s, %s, %s, %s, %s, %s)
            """
            
            # Extract risk level from analysis text
            analysis_text = analysis_data.get('analysis', '')
            risk_level = 'UNKNOWN'
            if 'Risk Level: HIGH' in analysis_text:
                risk_level = 'HIGH'
            elif 'Risk Level: MEDIUM' in analysis_text:
                risk_level = 'MEDIUM'
            elif 'Risk Level: LOW' in analysis_text:
                risk_level = 'LOW'
            
            cursor.execute(insert_query, (
                analysis_id,
                analysis_data.get('claim_id', str(uuid.uuid4())),
                analysis_data.get('claim_description'),
                risk_level,
                analysis_text,
                analysis_data.get('model', 'claude-3.5-sonnet')
            ))
            
            self.connection.commit()
            cursor.close()
            
            logger.info(f"Inserted Bedrock analysis {analysis_id} into Snowflake")
            return True
            
        except Exception as e:
            logger.error(f"Failed to insert Bedrock analysis: {str(e)}")
            return False
    
    def get_recent_claims(self, limit: int = 10) -> List[Dict]:
        """
        Get recent claims from Snowflake
        
        Args:
            limit: Number of claims to retrieve
        """
        try:
            if not self.connection:
                if not self.connect():
                    return []
            
            cursor = self.connection.cursor()
            
            query = f"""
            SELECT CLAIM_ID, CLAIM_DESCRIPTION, CLAIM_AMOUNT, 
                   CUSTOMER_ID, CLAIM_DATE, CREATED_AT
            FROM CLAIMS
            ORDER BY CREATED_AT DESC
            LIMIT {limit}
            """
            
            cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in cursor:
                results.append(dict(zip(columns, row)))
            
            cursor.close()
            logger.info(f"Retrieved {len(results)} claims from Snowflake")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get recent claims: {str(e)}")
            return []
    
    def get_high_risk_claims(self, limit: int = 10) -> List[Dict]:
        """
        Get high-risk claims based on Bedrock analysis
        
        Args:
            limit: Number of claims to retrieve
        """
        try:
            if not self.connection:
                if not self.connect():
                    return []
            
            cursor = self.connection.cursor()
            
            query = f"""
            SELECT ANALYSIS_ID, CLAIM_ID, CLAIM_DESCRIPTION, 
                   RISK_LEVEL, MODEL_NAME, CREATED_AT
            FROM BEDROCK_ANALYSIS
            WHERE RISK_LEVEL = 'HIGH'
            ORDER BY CREATED_AT DESC
            LIMIT {limit}
            """
            
            cursor.execute(query)
            
            columns = [desc[0] for desc in cursor.description]
            results = []
            
            for row in cursor:
                results.append(dict(zip(columns, row)))
            
            cursor.close()
            logger.info(f"Retrieved {len(results)} high-risk claims from Snowflake")
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get high-risk claims: {str(e)}")



