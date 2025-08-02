import numpy as np
import json
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# ✅ Load trained CNN model
model = load_model("basic_cnn_model.keras")  # Adjust path if needed

# ✅ Load class label mapping
with open("class_indices.json") as f:
    class_indices = json.load(f)

# ✅ Create reverse mapping for class index -> label
class_labels = [k for k, v in sorted(class_indices.items(), key=lambda item: item[1])]

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((224, 224))
    image = image.convert("RGB")
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_species(image: Image.Image) -> tuple:
    try:
        processed = preprocess_image(image)
        predictions = model.predict(processed)[0]
        predicted_index = np.argmax(predictions)
        predicted_label = class_labels[predicted_index]
        confidence = float(predictions[predicted_index]) * 100
        return predicted_label, confidence
    except Exception as e:
        print("Prediction error:", e)
        return None, None
