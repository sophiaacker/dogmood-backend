# main.py — classifier labels end-to-end
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import os, uuid, tempfile, subprocess

from snoutscout_classifier import dog_bark_classifier as clf
from suggestions import llm_suggestion, normalize_classifier_probs, CLASSIFIER_LABELS

print("🐶 Using classifier file:", clf.__file__)
import suggestions as _sug
print("🧠 Using suggestions file:", _sug.__file__)

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
app = FastAPI(title="DogMood API", version="0.2.0")

class AnalysisResult(BaseModel):
    label: str                                 # joy/boredom/hunger/aggressivity/sadness
    confidence: float
    probs: Optional[Dict[str, float]] = None   # classifier-space probs
    state: str
    suggestion: str
    reason: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"ok": True, "version": app.version}

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    # 1) save upload
    name = file.filename or "upload.bin"
    _, ext = os.path.splitext(name.lower())
    tmp_in = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext or '.bin'}")
    with open(tmp_in, "wb") as f:
        f.write(await file.read())

    # 2) force mono 16k WAV
    def _to_wav16k(src: str) -> str:
        out = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", out], check=True)
            return out
        except Exception:
            return src

    audio_path = tmp_in
    if ext in VIDEO_EXTS or ext != ".wav":
        audio_path = _to_wav16k(tmp_in)

    # 3) run classifier (classifier-space outputs)
    train_dir = os.path.dirname(os.path.abspath(clf.__file__))
    res = clf.classify_clip(audio_path, train_dir=train_dir, k=5)

    raw_label = str(res.get("prediction_label", "unknown")).strip().lower()
    raw_probs: Dict[str, float] = res.get("average_class_probabilities") or {}

    # 4) normalize & clamp to classifier label set
    probs = normalize_classifier_probs(raw_probs)
    if raw_label not in CLASSIFIER_LABELS:
        raw_label = max(probs, key=probs.get, default="unknown")

    confidence = float(probs.get(raw_label, 0.0))

    # 5) build scores in classifier space for suggester
    scores: List[dict] = sorted(
        ({"label": k, "score": float(v)} for k, v in probs.items()),
        key=lambda x: x["score"],
        reverse=True,
    ) if probs else []

    # 6) generate user-facing text (already in classifier space)
    merged = llm_suggestion(
        top_label=raw_label,
        scores=scores or None,
        context=f"Uploaded file: {name}",
        classifier_space=True,  # labels are already classifier labels
    )

    return AnalysisResult(
        label=raw_label,
        confidence=confidence,
        probs=probs or None,
        state=merged["state"],
        suggestion=merged["suggestion"],
        reason=merged["reason"],
    )
