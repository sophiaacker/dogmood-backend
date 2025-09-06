# main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import os, uuid, tempfile, subprocess

# local modules
import classifier  # import the module, not the symbol
from suggestions import llm_suggestion  # ⬅️ use the LLM, not the static table

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}

app = FastAPI(title="DogMood API", version="0.1.0")

# --- API models ---
class AnalysisResult(BaseModel):
    # classifier results
    label: str
    confidence: float
    probs: Optional[Dict[str, float]] = None

    # LLM merged, user-facing explanation
    state: str
    suggestion: str
    reason: str

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

    # 2) if video, try to extract mono 16k audio (ok if ffmpeg missing)
    audio_path = tmp_in
    if ext in VIDEO_EXTS:
        out = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", tmp_in, "-ac", "1", "-ar", "16000", out], check=True)
            audio_path = out
        except Exception:
            audio_path = tmp_in

    # 3) run your classifier
    result = classifier.classify_bark(audio_path)  # expected: {"label": "...", "confidence": 0.xx, "probs": {...}}
    label = str(result.get("label", "unknown"))
    confidence = float(result.get("confidence", 0.0))
    probs: Dict[str, float] = result.get("probs") or {}

    # convert probs -> ranked scores list for the LLM (highest first)
    scores: List[dict] = sorted(
        ({"label": k, "score": float(v)} for k, v in probs.items()),
        key=lambda x: x["score"],
        reverse=True,
    ) if probs else []

    # 4) build a tiny context (tweak as you like)
    context = f"Uploaded file: {name}"

    # 5) ask the LLM for the merged state/suggestion/reason
    merged = llm_suggestion(
        top_label=label,
        scores=scores or None,
        context=context,
    )
    # merged is: {"state": str, "suggestion": str, "reason": str}

    return AnalysisResult(
        label=label,
        confidence=confidence,
        probs=probs or None,
        state=merged["state"],
        suggestion=merged["suggestion"],
        reason=merged["reason"],
    )
