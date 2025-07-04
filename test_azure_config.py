import os
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

load_dotenv()

def test_azure_config():
    """Test Azure OpenAI configuration"""
    
    # Get environment variables
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    azure_deployment = os.getenv("DEPLOYMENT_NAME")
    api_version = os.getenv("API_VERSION")
    
    print("=== Azure OpenAI Configuration ===")
    print(f"API Key: {'*' * 10 if azure_api_key else 'NOT SET'}")
    print(f"Endpoint: {azure_endpoint}")
    print(f"Deployment: {azure_deployment}")
    print(f"API Version: {api_version}")
    print()
    
    # Check if all required variables are set
    if not all([azure_api_key, azure_endpoint, azure_deployment, api_version]):
        print("❌ ERROR: Missing required environment variables!")
        return False
    
    # Try different API versions
    api_versions_to_try = ["2024-02-15-preview", "2024-02-01", "2023-12-01-preview", "2023-09-01-preview"]
    
    for version in api_versions_to_try:
        print(f"Trying API version: {version}")
        try:
            # Initialize the LLM
            llm = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                openai_api_key=azure_api_key,
                deployment_name=azure_deployment,
                api_version=version,
                temperature=0.1,
            )
            
            # Test with a simple message
            system = SystemMessage(content="You are a helpful assistant.")
            user = HumanMessage(content="Say 'Hello, Azure OpenAI is working!'")
            
            print("Testing Azure OpenAI connection...")
            response = llm.invoke([system, user])
            print(f"✅ SUCCESS with API version {version}: {response.content}")
            return True
            
        except Exception as e:
            print(f"❌ FAILED with API version {version}: {str(e)}")
            continue
    
    print("\n❌ All API versions failed. Please check:")
    print("1. Deployment name is correct")
    print("2. Endpoint URL is correct")
    print("3. API key is valid")
    print("4. You have access to the deployment")
    return False

if __name__ == "__main__":
    test_azure_config() 