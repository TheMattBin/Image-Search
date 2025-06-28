import os
from PIL import Image
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.query import BM25Operator

load_dotenv()

weaviate_url = os.environ["WEAVIATE_URL"]
weaviate_api_key = os.environ["WEAVIATE_API_KEY"]

# 1. Initialize Weaviate client
# Ensure WEAVIATE_URL and WEAVIATE_API_KEY environment variables are set
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=f"https://{weaviate_url}",
    auth_credentials=Auth.api_key(api_key=weaviate_api_key),
)

# Check if Weaviate client is ready
if client.is_ready():
    print("Weaviate client connected successfully!")
else:
    print("Failed to connect to Weaviate. Please check your URL and API key.")
    exit() # Exit if connection fails


# --- Text Query Example ---
query_text = "cats"
try:
    images = client.collections.get("ImageObject")
    response = images.query.bm25(
        query=query_text,
        query_properties=["description"],
        operator=BM25Operator.or_(minimum_match=1),
        limit=1,
    )

    for o in response.objects:
        print(o.properties)

    client.close()

    # images = client.collections.get("ImageObject")
    # response = images.query.near_image(
    #     near_image=Path("../images/cats/4.jpg"),  # Provide a `Path` object
    #     # return_properties=["breed"],
    #     limit=1,
    #     # targetVector: "vector_name" # required when using multiple named vectors
    # )

    # print(response.objects[0])

    # client.close()

except NotImplementedError as e:
    print(f"Skipping text query: {e}")
except Exception as e:
    print(f"Error during text query: {e}")

# --- Image Query Example ---
# query_image_path = "path/to/your/query_image.jpg" # <--- IMPORTANT: Change this to your query image file
# if not os.path.exists(query_image_path):
#     print(f"Error: Query image file not found at {query_image_path}. Please update 'query_image_path'.")
# else:
#     try:
#         query_image = Image.open(query_image_path).convert("RGB")
#         image_query_embedding = extract_embedding(query_image)
#         print(f"Extracted embedding for query image with dimension: {len(image_query_embedding)}")

#         results_image_query = (
#             client.query.get("ImageObject", ["caption", "imageUrl"])
#             .with_near_vector({"vector": image_query_embedding})
#             .with_limit(3) # Get top 3 similar images
#             .do()
#         )
#         print(f"\nResults for image query (from {query_image_path}):")
#         if "data" in results_image_query and "Get" in results_image_query["data"] and "ImageObject" in results_image_query["data"]["Get"]:
#             for i, item in enumerate(results_image_query["data"]["Get"]["ImageObject"]):
#                 print(f"  Result {i+1}: Caption: '{item['caption']}', Image URL: '{item['imageUrl']}'")
#         else:
#             print("No results found for image query.")
#     except NotImplementedError as e:
#         print(f"Skipping image query: {e}")
#     except Exception as e:
#         print(f"Error during image query: {e}")
