import os
from typing import List, Optional
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth

class WeaviateImageDB:
    def __init__(self) -> None:
        load_dotenv()
        self.weaviate_url: str = os.environ["WEAVIATE_URL"]
        self.weaviate_api_key: str = os.environ["WEAVIATE_API_KEY"]
        self.client: weaviate.WeaviateClient = weaviate.connect_to_weaviate_cloud(
            cluster_url=f"https://{self.weaviate_url}",
            auth_credentials=Auth.api_key(api_key=self.weaviate_api_key),
        )
        if not self.client.is_ready():
            raise RuntimeError("Failed to connect to Weaviate.")

    def insert_image(self, image_id: str, embedding: List[float], caption: str, image_url: str = "") -> Optional[str]:
        imgObject = self.client.collections.get("ImageObject")
        data_object = {
            "description": caption,
            "imageUrl": image_url,
            "imageId": image_id,
        }
        return imgObject.data.insert(
            properties=data_object,
            vector=embedding,
            uuid=image_id
        )

    def query_by_vector(self, embedding: List[float], limit: int = 3):
        imgObject = self.client.collections.get("ImageObject")
        return imgObject.query.near_vector(
            near_vector=embedding,
            limit=limit
        )
    
    def query_by_text(self, text_query: str, limit: int = 3):
        imgObject = self.client.collections.get("ImageObject")
        return imgObject.query.bm25(
            query=text_query,
            query_properties=["description"],
            operator=weaviate.classes.query.BM25Operator.or_(minimum_match=1),
            limit=limit
        )

    def close(self) -> None:
        self.client.close()