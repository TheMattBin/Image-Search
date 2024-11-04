from pathlib import Path
import weaviate
import os
import cv2  # opencv-python package
import uuid
import base64


#TODO: Set up Weaviate client with OS environment variables
def weaviate_client():
    return weaviate.Client(
        url='https://xqi5m7uzruc8rzqp54hhew.c0.us-west3.gcp.weaviate.cloud',  # Replace with your Weaviate sandbox URL
        auth_client_secret=weaviate.auth.AuthApiKey(api_key='6g8CT2sc35tpHF3YwK21Y0zZpta9nbNr9nHt')  # Replace with your API key
    )

client = weaviate_client()

def define_schema(client):
    class_obj = {
       "class": "Pat",
       "description": "Pats with an image blob and a path to the image file",  
       "properties": [
            {
                "dataType": [
                    "blob"
                ],
                "description": "Image",
                "name": "image"
            },
            {
                "dataType": [
                    "string"
                ],
                "description": "",
                "name": "path"
            }
            
        ],
        "vectorIndexType": "hnsw",
        "moduleConfig": {
            "img2vec-neural": {
                "imageFields": [
                    "image"
                ]
            }
        },
        "vectorizer": "img2vec-neural"
    }
    
    client.schema.create_class(class_obj)

define_schema(client)


DATA_DIR = "processed_images"
IMAGE_DIM = (100, 100)

def _prepare_image(file_path):
    img = cv2.imread(file_path)
    # resize image
    resized = cv2.resize(img, IMAGE_DIM, interpolation= cv2.INTER_LINEAR)
    return resized

def insert_data(client):
    for file_name in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, file_name)
        img = _prepare_image(file_path)
        # encode image as base64 string
        jpg_img = cv2.imencode('.jpg', img)
        b64_string = base64.b64encode(jpg_img[1]).decode('utf-8')
        # define properties as expected by the class definition
        data_properties = {
            "path": file_name,
            "image": b64_string,
        }
        # create data of type "Pat" with a random UUID
        r = client.data_object.create(
            data_properties, 
            "Pat", 
            str(uuid.uuid4())
        )

insert_data(client)

# def search(client, test_image_path):
#     # prepare query and payload
#     near_image = {
#         "image": tmp_file_name,
#     }
#     query = client.query.get(
#         "Stamp",  # get data from class "Stamp"
#         ["path"]  # return the "path" property
#     )
#     .with_near_image(
#         near_image   # find "Stamp" close to img in embedding space
#     )
#     .with_limit(3)  # limit result to first 3 best matches
#     # perform query
#     res = query.do()
#     # return results
#     # (res is a dict following GraphQL syntax)
#     return res["data"]["Get"]["Stamp"]

# # read test image and transform it
# test_image = os.path.join(TEST_DIR, "photo.jpg")

# # save prepared image into a tmp file
# test_image_prepared = "test_image.jpg"
# cv2.imwrite(test_image_prepared, prepared)
# # perform query
# res = search(client, test_image_prepared)