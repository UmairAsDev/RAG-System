from qdrant_client import QdrantClient
import yaml

with open ('config/config.yaml', 'r') as file:
    secrets = yaml.safe_load(file)
    url = secrets["QDRANT_URL"]
    api_key = secrets["QDRANT_API"]
    
print(f"{url}...{api_key}")