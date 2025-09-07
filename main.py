# main.py — unified classifier for audio (bark) and image (skin) analysis
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Optional, List
import os, uuid, tempfile, subprocess, sys
import numpy as np
import joblib

# Import bark classifier
from snoutscout_classifier import dog_bark_classifier as bark_clf
from suggestions import llm_suggestion, normalize_classifier_probs, CLASSIFIER_LABELS, SKIN_LABELS, ALL_LABELS, SKIN_LABELS, ALL_LABELS

# Import image classifier
sys.path.append(os.path.join(os.path.dirname(__file__), 'dog_skin_classifier'))
try:
    from dog_skin_knn_fixed import extract_features
    SKIN_CLASSIFIER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Could not import skin classifier: {e}")
    SKIN_CLASSIFIER_AVAILABLE = False

print("🐶 Using bark classifier file:", bark_clf.__file__)
import suggestions as _sug
print("🧠 Using suggestions file:", _sug.__file__)

# File type definitions
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".aac", ".flac"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

app = FastAPI(title="DogMood API", version="0.3.0")

class AnalysisResult(BaseModel):
    analysis_type: str                         # "bark" or "skin"
    label: str                                 # bark: joy/boredom/etc, skin: ear/atopic/etc
    confidence: float
    probs: Optional[Dict[str, float]] = None   # classifier probabilities
    state: str
    suggestion: str
    products: Optional[List[str]] = None       # product recommendations
    reason: str

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load skin classifier model
SKIN_MODEL_PATH = os.path.join("dog_skin_classifier", "knn_skin.joblib")
try:
    if SKIN_CLASSIFIER_AVAILABLE:
        model_data = joblib.load(SKIN_MODEL_PATH)
        # The model file contains a dict with 'pipeline', 'labels', etc.
        skin_model = model_data['pipeline']  # Extract the actual sklearn pipeline
        skin_labels = model_data.get('labels', ['ear', 'atopic', 'acute', 'lick'])
        print(f"🔬 Loaded skin classifier: {SKIN_MODEL_PATH}")
    else:
        skin_model = None
        skin_labels = []
except Exception as e:
    print(f"⚠️ Could not load skin classifier: {e}")
    skin_model = None

@app.get("/health")
def health():
    return {
        "ok": True, 
        "version": app.version,
        "bark_classifier": "available",
        "skin_classifier": "available" if skin_model else "unavailable"
    }

def analyze_bark(file_path: str, filename: str) -> Dict:
    """Analyze dog bark audio"""
    # Convert to WAV if needed
    def _to_wav16k(src: str) -> str:
        out = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.wav")
        try:
            subprocess.run(["ffmpeg", "-y", "-i", src, "-ac", "1", "-ar", "16000", out], check=True, capture_output=True)
            return out
        except Exception:
            return src

    _, ext = os.path.splitext(filename.lower())
    audio_path = file_path
    if ext in VIDEO_EXTS or ext != ".wav":
        audio_path = _to_wav16k(file_path)

    # Run bark classifier
    train_dir = os.path.dirname(os.path.abspath(bark_clf.__file__))
    res = bark_clf.classify_clip(audio_path, train_dir=train_dir, k=5)

    raw_label = str(res.get("prediction_label", "unknown")).strip().lower()
    raw_probs: Dict[str, float] = res.get("average_class_probabilities") or {}

    # Normalize probabilities
    probs = normalize_classifier_probs(raw_probs)
    if raw_label not in CLASSIFIER_LABELS:
        raw_label = max(probs, key=probs.get, default="unknown")

    confidence = float(probs.get(raw_label, 0.0))

    # Build scores for suggestion system
    scores: List[dict] = sorted(
        ({"label": k, "score": float(v)} for k, v in probs.items()),
        key=lambda x: x["score"],
        reverse=True,
    ) if probs else []

    return {
        "analysis_type": "bark",
        "label": raw_label,
        "confidence": confidence,
        "probs": probs,
        "scores": scores,
        "context": f"Dog bark audio: {filename}"
    }

def analyze_skin(file_path: str, filename: str) -> Dict:
    """Analyze dog skin condition image"""
    if not skin_model or not SKIN_CLASSIFIER_AVAILABLE:
        raise HTTPException(status_code=503, detail="Skin classifier not available")

    try:
        # Load and preprocess the image first
        from PIL import Image
        import numpy as np
        
        # Load image and convert to RGB array
        with Image.open(file_path) as img:
            img_rgb = img.convert('RGB')
            img_rgb = img_rgb.resize((256, 256))  # Match training size
            img_array = np.array(img_rgb) / 255.0  # Normalize to [0,1] as expected by extract_features
        
        # Extract features using the same method as the training
        features = extract_features(img_array)
        
        # Get prediction and probabilities
        prediction = skin_model.predict([features])[0]
        
        # Try to get probabilities if available
        try:
            probabilities = skin_model.predict_proba([features])[0]
            # Use the actual model classes (which may be different from stored labels)
            model_classes = skin_model.classes_
            probs = {model_classes[i]: float(probabilities[i]) for i in range(len(model_classes))}
            confidence = float(probabilities[np.argmax(probabilities)])
        except AttributeError:
            # Fallback if no predict_proba available
            probs = {prediction: 1.0}
            confidence = 1.0

        # Build scores for suggestion system
        scores: List[dict] = sorted(
            ({"label": k, "score": float(v)} for k, v in probs.items()),
            key=lambda x: x["score"],
            reverse=True,
        ) if probs else []

        return {
            "analysis_type": "skin",
            "label": prediction,
            "confidence": confidence,
            "probs": probs,
            "scores": scores,
            "context": f"Dog skin condition image: {filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error analyzing skin image: {str(e)}")

@app.post("/analyze", response_model=AnalysisResult)
async def analyze(file: UploadFile = File(...)):
    # 1) Save uploaded file
    name = file.filename or "upload.bin"
    _, ext = os.path.splitext(name.lower())
    tmp_in = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext or '.bin'}")
    
    with open(tmp_in, "wb") as f:
        f.write(await file.read())

    try:
        # 2) Determine file type and route to appropriate classifier
        if ext in AUDIO_EXTS or ext in VIDEO_EXTS:
            # Audio/Video -> Bark Analysis
            result = analyze_bark(tmp_in, name)
        elif ext in IMAGE_EXTS:
            # Image -> Skin Analysis
            result = analyze_skin(tmp_in, name)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Supported: audio ({', '.join(AUDIO_EXTS)}), video ({', '.join(VIDEO_EXTS)}), image ({', '.join(IMAGE_EXTS)})")

        # 3) Generate suggestions using LLM
        merged = llm_suggestion(
            top_label=result["label"],
            scores=result.get("scores"),
            context=result["context"],
            classifier_space=True,
        )

        return AnalysisResult(
            analysis_type=result["analysis_type"],
            label=result["label"],
            confidence=result["confidence"],
            probs=result.get("probs"),
            state=merged["state"],
            suggestion=merged["suggestion"],
            products=merged.get("products"),
            reason=merged["reason"],
        )

    finally:
        # Cleanup temporary files
        try:
            os.unlink(tmp_in)
        except:
            pass
