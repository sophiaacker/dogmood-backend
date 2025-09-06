from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional
import os, uuid, tempfile, subprocess

# local modules
from suggestions import SUGGESTIONS
import classifier  # import the module, not the symbol

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}

app = FastAPI(title="DogMood API", version="0.1.0")

class AnalysisResult(BaseModel):
    label: str
    confidence: float
    probs: Optional[Dict[str, float]] = None
    suggestion: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"]
)

@app.get("/health")
def health():
    return {"ok": True, "version": "0.1.0"}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    # 1) save upload to a temp file
    name = file.filename or "upload.bin"
    _, ext = os.path.splitext(name.lower())
    tmp_in = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext or '.bin'}")
    with open(tmp_in, "wb") as f:
        f.write(await file.read())

    # 2) if video, try to extract mono 16k audio (fine if ffmpeg missing; mock doesn’t need it)
    audio_path = tmp_in
    if ext in VIDEO_EXTS:
        out = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", tmp_in, "-ac", "1", "-ar", "16000", out], check=True)
            audio_path = out
        except Exception:
            # no ffmpeg? fall back; mock doesn't care
            audio_path = tmp_in

    # 3) run mock classifier
    result = classifier.classify_bark(audio_path)
    label = result.get("label", "unknown")
    confidence = float(result.get("confidence", 0.0))
    suggestion = SUGGESTIONS.get(label, SUGGESTIONS["unknown"])
    return AnalysisResult(label=label, confidence=confidence, probs=result.get("probs"), suggestion=suggestion)