import os
from PIL import Image

def convert_images_to_jpg(directory):
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            try:
                with Image.open(filepath) as img:
                    rgb_img = img.convert('RGB')
                    new_filename = os.path.splitext(filename)[0] + '.jpg'
                    new_filepath = os.path.join(directory, new_filename)
                    rgb_img.save(new_filepath, 'JPEG')
            except Exception as e:
                print(f"Skipping {filename}: {e}")

# Usage
convert_images_to_jpg('assets')