from weaviate_db import WeaviateImageDB
from vision_models import VisionModels

db = WeaviateImageDB()
vision = VisionModels()

# For indexing
image_path = "path/to/image.jpg"
caption = vision.generate_caption(image_path)
embedding = vision.extract_embedding(image_path)
db.insert_image("unique-id", embedding, caption, image_url="optional_url")

# For querying
results = db.query_by_vector(embedding)
for obj in results.objects:
    print(obj.properties)

results = db.query_by_text("Query text here")
for obj in results.objects:
    print(obj.properties)

db.close()