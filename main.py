from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
import io
import json
import os

app = FastAPI()

MODEL_PATH = "./best_model.h5"
CLASS_NAMES_PATH = "class_names.json"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

if os.path.exists(CLASS_NAMES_PATH):
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = json.load(f)
else:
    raise RuntimeError(f"Class names file not found at {CLASS_NAMES_PATH}. Please provide it.")


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((128, 128))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        input_tensor = preprocess_image(image)

        predictions = model.predict(input_tensor)
        pred_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][pred_index])
        label = class_names[pred_index]

        return JSONResponse({
            "label": label,
            "confidence": round(confidence, 4),
            "class_index": pred_index
        })

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    except Exception as e:
        print(f"ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
