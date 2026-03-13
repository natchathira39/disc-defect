import os
import gdown
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI(title="Disc Brake Defect Detection API")

MODEL_PATH = "disc_brake_final_fixed.h5"

MODEL_URL = "MODEL_URL = "https://drive.google.com/uc?id=1o7V9fmgQzCVRct1TOaZb8D5-170_YkQH""

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
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
    return {"message": "Disc Brake Defect Detection API Running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")

    img = preprocess_image(image)

    prediction = model.predict(img)

    prob = float(prediction[0][0])

    label = "Defective Disc Brake" if prob > 0.5 else "Normal Disc Brake"

    return {
        "prediction": label,
        "confidence": prob
    }
