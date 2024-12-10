import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from PIL import Image as PILImage
from huggingface_hub import hf_hub_download, from_pretrained_keras

# Download the test image from Hugging Face Hub
hf_hub_download(repo_id="google/path-foundation", filename='Test.png', local_dir='.')

# Open the image, crop it, convert it to RGB format, and display it.
img = PILImage.open("Test.png").crop((0, 0, 224, 224)).convert('RGB')

# Convert the image to a Tensor and scale to [0, 1]
tensor = tf.cast(tf.expand_dims(np.array(img), axis=0), tf.float32) / 255.0

# Load the model directly from Hugging Face
loaded_model = from_pretrained_keras("google/path-foundation")

# Call inference
infer = loaded_model.signatures["serving_default"]
embeddings = infer(tf.constant(tensor))

# Extract the embedding vector
embedding_vector = embeddings['output_0'].numpy().flatten()
print("Size of embedding vector:", len(embedding_vector))

# Plot the embedding vector
plt.figure(figsize=(12, 4))
plt.plot(embedding_vector)
plt.title('Embedding Vector')
plt.xlabel('Index')
plt.ylabel('Value')
plt.grid(True)
plt.show()
