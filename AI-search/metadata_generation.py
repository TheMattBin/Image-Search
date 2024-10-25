import requests
import os
import json

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large-ft", torch_dtype=torch_dtype, trust_remote_code=True).to(device)
processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large-ft", trust_remote_code=True)

def run_example(task_prompt, text_input=None, image=None):
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input
    inputs = processor(text=prompt, images=image, return_tensors="pt").to(device, torch_dtype)
    generated_ids = model.generate(
      input_ids=inputs["input_ids"],
      pixel_values=inputs["pixel_values"],
      max_new_tokens=1024,
      num_beams=3
    )
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

    parsed_answer = processor.post_process_generation(generated_text, task=task_prompt, image_size=(image.width, image.height))

    print(parsed_answer)


prompt = "<MORE_DETAILED_CAPTION>"
text_input = "Extract metadata, such as image_id, image_title, image_description,image_date, image_resolution, \
    image_orientation, image_tags, image_keywords, image_file_creation_date, and formatted in json."

# url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg?download=true"
# image = Image.open(requests.get(url, stream=True).raw)

images_metadata = []

for image in os.listdir("images"):
    image = Image.open(f"images/{image}").convert("RGB")
    metadata = run_example(prompt, text_input, image)
    images_metadata.append(metadata)

"""
Example of metadata generated for an image:
{
  "image_id": "001",
  "image_title": "Cats in Sink",
  "image_description": "A group of cats is lying in a bathroom sink.",
  "image_date": "2023-04-01",
  "image_resolution": "1920x1080",
  "image_orientation": "portrait",
  "image_tags": ["cats", "bathroom", "sink", "domestic animals", "pets"],
  "image_keywords": ["cats", "bathroom", "sink", "pets", "domestic animals"],
  "image_file_creation_date": "2023-04-01"
}
"""


# Write the list of images to a JSON file
with open('images_metadata.json', 'w') as json_file:
    json.dump(images_metadata, json_file, indent=4)

print("Data has been written to images.json")