from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import numpy as np
import io
import os

from flask_cors import CORS  #

# Load model dan label encoder
model = load_model("model/model_xray.h5")
labels = ["Atelectasis", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumothorax"]

app = Flask(__name__)
CORS(app)  # ✅ Aktifkan CORS agar bisa diakses dari frontend lain (misal Netlify)

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L")  # Grayscale
    image = image.resize((224, 224))
    image = np.array(image).astype('float32') / 255.0
    image = np.stack([image] * 3, axis=-1)  # Grayscale to RGB
    image = np.expand_dims(image, axis=0)
    return image

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image_bytes = request.files['image'].read()
    image = preprocess_image(image_bytes)
    prediction = model.predict(image)[0]

    idx = np.argmax(prediction)
    confidence = float(prediction[idx]) * 100

    return jsonify({
        "condition": labels[idx],
        "confidence": f"{confidence:.2f}%"
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
