from fastapi import FastAPI, File, UploadFile, HTTPException
import joblib
import cv2
import numpy as np
import pandas as pd
from rembg import remove
from io import BytesIO
from PIL import Image

app = FastAPI()

# Load trained model
model = joblib.load("model.pkl")

# Feature extraction function
def extract_features_opencv(image_path):
    img_bgr_alpha = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img_bgr_alpha is None or img_bgr_alpha.shape[2] < 3:
        return None
    if img_bgr_alpha.shape[2] == 4:
        bgr = img_bgr_alpha[:, :, :3]
        alpha = img_bgr_alpha[:, :, 3]
    else:
        bgr = img_bgr_alpha
        alpha = np.ones(bgr.shape[:2], dtype=np.uint8) * 255
    mask = alpha > 0
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)[mask]
    if rgb.size == 0:
        return None
    mean_R = np.mean(rgb[:, 0])
    mean_G = np.mean(rgb[:, 1])
    mean_B = np.mean(rgb[:, 2])
    sum_rgb = mean_R + mean_G + mean_B
    nr = mean_R / sum_rgb if sum_rgb else 0
    ng = mean_G / sum_rgb if sum_rgb else 0
    nb = mean_B / sum_rgb if sum_rgb else 0
    gmr = mean_G / mean_R if mean_R else 0
    gmb = mean_G / mean_B if mean_B else 0
    bgr_pixels = bgr[mask]
    lab = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2LAB).reshape(-1, 3)
    hsv = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    ycbcr = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2YCrCb).reshape(-1, 3)
    L, a, b = np.mean(lab[:, 0]), np.mean(lab[:, 1]), np.mean(lab[:, 2])
    H_mean, S, Y_mean = np.mean(hsv[:, 0]), np.mean(hsv[:, 1]), np.mean(ycbcr[:, 0])
    gdr = (mean_G - mean_R) / (mean_G + mean_R) if (mean_G + mean_R) else 0
    VI = (2 * mean_G - mean_R - mean_B) / (2 * mean_G + mean_R + mean_B) if (2 * mean_G + mean_R + mean_B) else 0
    return pd.DataFrame([{
        'R': mean_R, 'G': mean_G, 'B': mean_B,
        'nr': nr, 'ng': ng, 'nb': nb,
        'gmr': gmr, 'gmb': gmb,
        'L': L, 'a': a, 'b': b,
        'gdr': gdr, 'H_mean': H_mean, 'Y_mean': Y_mean, 'S': S, 'VI': VI
    }])

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Remove background
        input_bytes = await file.read()
        output_bytes = remove(input_bytes)
        img_no_bg = Image.open(BytesIO(output_bytes))
        image_path = "temp.png"
        img_no_bg.save(image_path)

        # Extract features
        features_df = extract_features_opencv(image_path)
        if features_df is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Predict
        prediction = model.predict(features_df)[0]
        return {"predicted_spad": float(prediction)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
