import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings

load_dotenv()

def test_azure_embeddings():
    """Test Azure OpenAI embeddings configuration"""
    
    # Get environment variables
    azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    embedding_deployment = os.getenv("EMBEDDING_DEPLOYMENT_NAME")
    embedding_model = os.getenv("EMBEDDING_MODEL_NAME")
    api_version = os.getenv("API_VERSION")
    
    print("=== Azure OpenAI Embeddings Configuration ===")
    print(f"API Key: {'*' * 10 if azure_api_key else 'NOT SET'}")
    print(f"Endpoint: {azure_endpoint}")
    print(f"Embedding Deployment: {embedding_deployment}")
    print(f"Embedding Model: {embedding_model}")
    print(f"API Version: {api_version}")
    print()
    
    # Check if all required variables are set
    if not all([azure_api_key, azure_endpoint, embedding_deployment, api_version]):
        print("❌ ERROR: Missing required environment variables!")
        return False
    
    try:
        # Initialize Azure embeddings
        embeddings = AzureOpenAIEmbeddings(
            azure_deployment=embedding_deployment,
            openai_api_key=azure_api_key,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
            chunk_size=1
        )
        
        # Test with a simple text
        test_text = "Hello, this is a test for Azure OpenAI embeddings!"
        
        print("Testing Azure OpenAI embeddings...")
        embedding = embeddings.embed_query(test_text)
        
        print(f"✅ SUCCESS: Generated embedding with {len(embedding)} dimensions")
        print(f"First 5 values: {embedding[:5]}")
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {str(e)}")
        return False

if __name__ == "__main__":
    test_azure_embeddings() 