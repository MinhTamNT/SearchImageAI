import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity

# Load the embeddings and image paths from the .npz file
output_file = "D://Studying//02-Self//SearchImage//SearchImage//image_embeddings.npz"
data = np.load(output_file)

loaded_embeddings = data['embeddings']
loaded_image_paths = data['image_paths']

query_embedding = loaded_embeddings[1]

query_image_path = loaded_image_paths[1]
query_image = Image.open(query_image_path)

plt.figure(figsize=(5, 5))
plt.imshow(query_image)
plt.title("Query Image")
plt.axis('off')
query_image.close()
plt.show()

cosine_similarities = cosine_similarity([query_embedding], loaded_embeddings)[0]

similarity_scores = list(enumerate(cosine_similarities))
similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

top_k = 5
plt.figure(figsize=(20, 20))

for i, (idx, score) in enumerate(similarity_scores[:top_k]):
    image_path = loaded_image_paths[idx]
    image = Image.open(image_path)

    # Hiển thị ảnh
    plt.subplot(1, top_k, i + 1)
    plt.imshow(image)
    plt.title(f"Similarity: {score:.4f}")
    plt.axis('off')
    image.close()

plt.show()
