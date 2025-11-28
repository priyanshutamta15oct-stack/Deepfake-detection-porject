# app.py
import shutil
import uuid
import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from model import load_model, preprocess, DEVICE
from utils import extract_sample_frames, aggregate_predictions
import torch
from PIL import Image

# Single app instance
app = FastAPI(title="Deepfake Detector Demo")

# CORS - allow only the dev origin (safer than "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # front-end origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model = load_model()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Basic validation
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    # Save uploaded file to a secure temp path
    suffix = os.path.splitext(file.filename)[1] or ".mp4"
    tmp_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}{suffix}")
    with open(tmp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        frames = extract_sample_frames(tmp_path, max_frames=16, stride=10)
        if not frames:
            raise HTTPException(status_code=400, detail="Could not extract frames from video")

        preds = []
        for frame in frames:
            im_pil = Image.fromarray(frame)  # utils returns RGB arrays
            tensor = preprocess(im_pil).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = model(tensor)
                prob = torch.nn.functional.softmax(out, dim=1).cpu().numpy()[0].tolist()
            preds.append(prob)

        score_fake = aggregate_predictions(preds)
        verdict = "FAKE" if score_fake > 0.4 else "REAL"
        return JSONResponse({"verdict": verdict, "fake_score": score_fake, "per_frame": preds})

    finally:
        # remove temp file
        try:
            os.remove(tmp_path)
        except Exception:
            pass
