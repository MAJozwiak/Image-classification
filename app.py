from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from io import BytesIO
from PIL import Image
import asyncio

app = FastAPI()

try:
    model = load_model("model.h5")
except Exception:
    model = None

class_names = ['cat', 'dog']


def prepare_image(img: Image.Image, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return JSONResponse(content={"error": "Model nie jest załadowany"}, status_code=400)

    try:
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB")
        img_array = prepare_image(img)
        preds = model.predict(img_array)
        predicted_class = class_names[np.argmax(preds)]
        confidence = float(np.max(preds))
        return {"class": predicted_class, "confidence": confidence}
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@app.post("/train")
async def train_model_endpoint():
    from train import train_model

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, train_model)

    global model
    model = load_model("model.h5")

    return {"status": "Model wytrenowany i załadowany", "train_result": result}


@app.post("/test")
async def test_model_endpoint():
    from train import test_model

    accuracy = test_model()
    return {"accuracy": accuracy}
