import tensorflow as tf
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import io

app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model("model.h5")

IMG_SIZE = 224   # change if your model used a different input size

def preprocess_image(image):
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.get("/")
def home():
    return {"message": "PCB Defect Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    processed = preprocess_image(image)

    prediction = model.predict(processed)

    probability = float(prediction[0][0])

    if probability > 0.5:
        label = "Defect"
    else:
        label = "No Defect"

    return {
        "prediction": label,
        "confidence": probability
    }
