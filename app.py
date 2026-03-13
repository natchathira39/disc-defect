import os
import gdown
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

MODEL_PATH = "model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1tCUvD3iEbWZU4UhijOa2kBXO78Xa0VDt"

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

IMG_SIZE = 128


def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


@app.get("/")
def home():
    return {"message": "Disc Defect Detection API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = preprocess_image(image)

    prediction = model.predict(img)

    prob = float(prediction[0][0])

    label = "Defective Disc" if prob > 0.5 else "Normal Disc"

    return {
        "prediction": label,
        "confidence": prob
    }
