import numpy as np
import json
import tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image as keras_image

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os

app = FastAPI()

# Allow frontend (running on a different port/origin) to call this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rebuild model architecture (same as training) and load trained weights
base_model_tf = ResNet50(include_top=False, weights='imagenet', input_shape=(224,224,3), classes=38)
base_model_tf.trainable = False

pt = Input(shape=(224,224,3))
func = keras.layers.Lambda(lambda x: tensorflow.cast(x, tensorflow.float32))(pt)
x = preprocess_input(func)
model_resnet = base_model_tf(x, training=False)
model_resnet = GlobalAveragePooling2D()(model_resnet)
model_resnet = Dense(128, activation='relu')(model_resnet)
model_resnet = Dense(64, activation='relu')(model_resnet)
model_resnet = Dense(38, activation='softmax')(model_resnet)

model_main = Model(inputs=pt, outputs=model_resnet)
model_main.load_weights('model/weights/RESNET50_FINETUNED.weights.h5')

# Load class index mapping
with open('model/weights/class_indices.json') as f:
    idx_to_class = json.load(f)  # keys are strings, e.g. {"0": "Apple___Apple_scab", ...}

def predict(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = np.expand_dims(keras_image.img_to_array(img), axis=0)

    predictions = model_main.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])
    class_name = idx_to_class[str(predicted_index)]

    return class_name, confidence

# API endpoints
@app.get("/")
def health_check():
    return {"status": "AgriSmart AI backend is running"}

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    class_name, confidence = predict(temp_path)

    os.remove(temp_path)

    return {
        "disease": class_name,
        "confidence": round(confidence, 4)
    }