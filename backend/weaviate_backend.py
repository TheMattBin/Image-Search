import weaviate
import os

client = weaviate.Client(
    url=f"https://{os.getenv('WEAVIATE_URL')}",
    auth_client_secret=weaviate.AuthApiKey(api_key=os.getenv('WEAVIATE_API_KEY')),
)

# Check if client is ready
if client.is_ready():
    print("Weaviate client connected successfully!")
else:
    print("Failed to connect to Weaviate.")
