import requests
import os

# Create directories for storing images
os.makedirs('images/dogs', exist_ok=True)

# Function to download images
def download_images(api_url, folder_name, num_images):
    for i in range(num_images):
        response = requests.get(api_url)
        if response.status_code == 200:
            # Access the first item in the list returned by the API
            image_data = response.json()[0]  # Get the first item from the list
            image_url = image_data['url']     # Now access the 'url' key
            img_data = requests.get(image_url).content
            with open(f'images/{folder_name}/{i + 1}.jpg', 'wb') as handler:
                handler.write(img_data)
            print(f'Downloaded {folder_name}/{i + 1}.jpg')
        else:
            print(f'Failed to fetch image from {api_url}')

# Download 10 cat images
download_images('https://api.thecatapi.com/v1/images/search', 'dogs', 10)