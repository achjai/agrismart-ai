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

# Rebuild model architecture (same as training) and load fine-tuned weights 
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

# Precaution / advice lookup 
PRECAUTIONS = {
    "Apple___Apple_scab": "Remove and destroy fallen leaves. Apply a fungicide in early spring before symptoms appear.",
    "Apple___Black_rot": "Prune out dead or diseased wood. Remove mummified fruit from the tree and ground.",
    "Apple___Cedar_apple_rust": "Remove nearby cedar/juniper trees if possible, or apply fungicide starting at bud break.",
    "Apple___healthy": "No action needed. Continue regular monitoring.",
    "Blueberry___healthy": "No action needed. Continue regular monitoring.",
    "Cherry_(including_sour)___Powdery_mildew": "Improve air circulation by pruning. Apply sulfur-based fungicide if severe.",
    "Cherry_(including_sour)___healthy": "No action needed. Continue regular monitoring.",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Rotate crops and avoid dense planting. Consider fungicide if infection is widespread.",
    "Corn_(maize)___Common_rust_": "Plant resistant varieties in future seasons. Fungicide is rarely needed unless severe.",
    "Corn_(maize)___Northern_Leaf_Blight": "Rotate crops, remove infected debris after harvest, consider resistant hybrids.",
    "Corn_(maize)___healthy": "No action needed. Continue regular monitoring.",
    "Grape___Black_rot": "Remove mummified berries and infected leaves. Apply fungicide starting early in the season.",
    "Grape___Esca_(Black_Measles)": "Remove and destroy infected wood. No effective chemical cure; focus on prevention.",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Improve air circulation via pruning. Apply fungicide if recurring.",
    "Grape___healthy": "No action needed. Continue regular monitoring.",
    "Orange___Haunglongbing_(Citrus_greening)": "Remove and destroy infected trees promptly to prevent spread. Control psyllid insects.",
    "Peach___Bacterial_spot": "Avoid overhead irrigation. Apply copper-based bactericide during dormancy.",
    "Peach___healthy": "No action needed. Continue regular monitoring.",
    "Pepper,_bell___Bacterial_spot": "Avoid overhead watering. Remove and destroy infected plants to limit spread.",
    "Pepper,_bell___healthy": "No action needed. Continue regular monitoring.",
    "Potato___Early_blight": "Remove affected leaves. Rotate crops and avoid overhead watering.",
    "Potato___Late_blight": "Remove and destroy infected plants immediately. Avoid overhead watering; apply fungicide preventatively in humid conditions.",
    "Potato___healthy": "No action needed. Continue regular monitoring.",
    "Raspberry___healthy": "No action needed. Continue regular monitoring.",
    "Soybean___healthy": "No action needed. Continue regular monitoring.",
    "Squash___Powdery_mildew": "Improve air circulation. Apply sulfur or potassium bicarbonate-based fungicide.",
    "Strawberry___Leaf_scorch": "Remove infected leaves after harvest. Avoid overhead watering.",
    "Strawberry___healthy": "No action needed. Continue regular monitoring.",
    "Tomato___Bacterial_spot": "Avoid overhead watering. Remove and destroy infected plant debris.",
    "Tomato___Early_blight": "Remove affected lower leaves. Avoid overhead watering; apply fungicide if recurring.",
    "Tomato___Late_blight": "Remove and destroy infected plants immediately - this spreads fast. Avoid overhead watering.",
    "Tomato___Leaf_Mold": "Improve greenhouse/field ventilation. Avoid overhead watering.",
    "Tomato___Septoria_leaf_spot": "Remove affected lower leaves. Rotate crops and avoid overhead watering.",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Increase humidity around plants. Consider insecticidal soap or neem oil for heavy infestations.",
    "Tomato___Target_Spot": "Remove affected leaves. Improve air circulation and avoid overhead watering.",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Remove and destroy infected plants. Control whiteflies, which spread this virus.",
    "Tomato___Tomato_mosaic_virus": "Remove and destroy infected plants. Wash hands/tools after handling to avoid spreading.",
    "Tomato___healthy": "No action needed. Continue regular monitoring.",
}

DEFAULT_PRECAUTION = "Consult a local agricultural expert for specific guidance on this condition."


def get_precaution(disease_name: str) -> str:
    return PRECAUTIONS.get(disease_name, DEFAULT_PRECAUTION)


def predict(image_path):
    img = keras_image.load_img(image_path, target_size=(224, 224))
    img_array = np.expand_dims(keras_image.img_to_array(img), axis=0)

    predictions = model_main.predict(img_array)
    predicted_index = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_index])
    class_name = idx_to_class[str(predicted_index)]

    return class_name, confidence


# ---- API endpoints ----
@app.get("/")
def health_check():
    return {"status": "AgriSmart AI backend is running"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    class_name, confidence = predict(temp_path)
    precaution = get_precaution(class_name)

    os.remove(temp_path)

    response = {
        "disease": class_name,
        "confidence": round(confidence, 4),
        "precaution": precaution
    }

    if confidence < 0.5:
        response["warning"] = "Low confidence - try a clearer, front-facing photo of the leaf in good lighting."

    return response