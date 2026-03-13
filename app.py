from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
from PIL import Image
import numpy as np
import io
import gdown
import os
import json

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH         = "disc_brake_deploy_final.h5"
CLASS_INDICES_PATH = "class_indices.json"
MODEL_URL          = "https://drive.google.com/uc?id=1thCV6xlGiW5hPfOom1qRnuRbrZ-Ypym5"
CLASS_INDICES_URL  = "https://drive.google.com/uc?id=1b1KHPfPde4v_PU6LPChw4WUYSJeJzKye"

def load_resources():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        print("Model downloaded!")

    if not os.path.exists(CLASS_INDICES_PATH):
        print("Downloading class indices from Google Drive...")
        gdown.download(CLASS_INDICES_URL, CLASS_INDICES_PATH, quiet=False)
        print("Class indices downloaded!")

    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded successfully!")

    with open(CLASS_INDICES_PATH, 'r') as f:
        class_indices = json.load(f)
    idx_to_class = {v: k for k, v in class_indices.items()}
    print("Class indices loaded!")

    return model, idx_to_class

model, idx_to_class = load_resources()

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Disc Brake Defect Detection API",
        "version": "1.0",
        "classes": list(idx_to_class.values())
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        if img.mode != 'RGB':
            img = img.convert('RGB')

        img       = img.resize((128, 128))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = model.predict(img_array, verbose=0)[0]
        pred_idx    = int(np.argmax(predictions))
        pred_class  = idx_to_class[pred_idx]
        confidence  = float(predictions[pred_idx]) * 100

        all_probs = {
            idx_to_class[i]: round(float(predictions[i]) * 100, 2)
            for i in range(len(predictions))
        }

        return {
            "prediction":        pred_class,
            "confidence":        round(confidence, 2),
            "is_defective":      pred_class != "undefective",
            "all_probabilities": all_probs
        }

    except Exception as e:
        return {
            "error":      str(e),
            "prediction": "ERROR",
            "confidence": 0
        }

@app.get("/health")
def health_check():
    return {
        "status":       "healthy",
        "model_loaded": model is not None,
        "classes":      list(idx_to_class.values())
    }
