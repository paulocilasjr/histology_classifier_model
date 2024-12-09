from PIL import Image as PILImage
from huggingface_hub import hf_hub_download
import tensorflow as tf
import numpy as np
from keras.layers import TFSMLayer

# Download a test image from Hugging Face Hub
hf_hub_download(repo_id="google/path-foundation", filename="Test.png", local_dir=".")

# Open the image, crop it to match expected input size (224x224), and convert to RGB
img = PILImage.open("Test.png").crop((0, 0, 224, 224)).convert("RGB")

# Convert the image to a Tensor and scale pixel values to [0, 1]
tensor = tf.cast(tf.expand_dims(np.array(img), axis=0), tf.float32) / 255.0

# Load the TensorFlow SavedModel using TFSMLayer
saved_model_path = (
    "/Users/4475918/.cache/huggingface/hub/models--google--path-foundation/snapshots/fd6a835ceaae15be80db6abd8dcfeb86a9287e72"
)
model = TFSMLayer(saved_model_path, call_endpoint="serving_default")

# Call inference
embeddings = model(tensor)

# Extract the embedding vector
embedding_vector = embeddings.numpy().flatten()

print("Embedding vector shape:", embedding_vector.shape)
