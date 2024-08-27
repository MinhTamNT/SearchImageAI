import os
from glob import glob
from PIL import Image
import matplotlib.pyplot as plt
import random
import numpy as np
from sentence_transformers import SentenceTransformer

# Initialize the model
model = SentenceTransformer("clip-VIT-B-32")

# Set the image folder path
images_folder = "D://Studying//02-Self//SearchImage//SearchImage//Dataset//archive//flickr30k_images//flickr30k_images"
image_file = glob(os.path.join(images_folder, "*.jpg"))

selected_files = random.sample(image_file, 5)

plt.figure(figsize=(50, 50))
for i, file in enumerate(selected_files):
    image = Image.open(file)
    plt.subplot(1, 5, i + 1)
    plt.imshow(image)
    plt.axis('off')
    image.close()
plt.show()

chunk_size = 128

embeddings = []

def process_chunk(chunk):
    images = []
    for image_file in chunk:
        images.append(Image.open(image_file))
    chunk_embeddings = model.encode(images)
    # Close the images after processing to free up memory
    for img in images:
        img.close()
    return chunk_embeddings

# Process images in chunks
for i in range(0, len(image_file), chunk_size):
    print(f"Processing chunk {i // chunk_size + 1}")
    chunk = image_file[i:i + chunk_size]
    chunk_embeddings = process_chunk(chunk)
    embeddings.extend(chunk_embeddings)

embeddings_array = np.array(embeddings)

output_file = "image_embeddings.npz"
np.savez(output_file, embeddings=embeddings_array, image_paths=image_file)

print(f"Embeddings đã được lưu vào {output_file}")



