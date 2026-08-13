
from PIL import Image, ImageOps
import numpy as np
import tf_keras as tk

np.set_printoptions(suppress=True)

# Load the trained model
model = tk.models.load_model("keras_model.h5", compile=False)

# Load class labels
class_names = open("labels.txt", "r").readlines()

# Create input array
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Load test image
image = Image.open("test_led.jpg").convert("RGB")

# Resize and crop image
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# Convert image to numpy array
image_array = np.asarray(image)

# Normalize image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Add image to model input
data[0] = normalized_image_array

# Make prediction
prediction = model.predict(data)
index = np.argmax(prediction)

class_name = class_names[index]
confidence_score = prediction[0][index]

# Display result
print("\n------------------------------")
print("Image Classification Result")
print("------------------------------")
print("Predicted Class :", class_name[2:].strip())
print("Confidence      :", f"{confidence_score * 100:.2f}%")
print("------------------------------")
