from dotenv import load_dotenv
import os

load_dotenv()

print("Checking Snowflake environment variables:")
print(f"SNOWFLAKE_ACCOUNT: {os.getenv('SNOWFLAKE_ACCOUNT')}")
print(f"SNOWFLAKE_USER: {os.getenv('SNOWFLAKE_USER')}")
print(f"SNOWFLAKE_PASSWORD: {'*' * len(os.getenv('SNOWFLAKE_PASSWORD', '')) if os.getenv('SNOWFLAKE_PASSWORD') else 'NOT SET'}")
print(f"SNOWFLAKE_WAREHOUSE: {os.getenv('SNOWFLAKE_WAREHOUSE')}")
print(f"SNOWFLAKE_DATABASE: {os.getenv('SNOWFLAKE_DATABASE')}")

# Try connecting
from src.api.snowflake_service import SnowflakeService

sf = SnowflakeService()
if sf.connect():
    print("\n✅ Successfully connected to Snowflake!")
    sf.disconnect()
else:
    print("\n❌ Failed to connect to Snowflake")