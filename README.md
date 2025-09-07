# 🐕 SnoutScout Backend

A unified FastAPI backend that analyzes dog barks and skin conditions to provide actionable care recommendations powered by AI.

USE BRANCH SNOUTSCOUT-REWIRE

## ✨ Features

### 🔊 **Bark Analysis**
- **Audio Classification**: Analyzes dog barks to detect emotional states
- **Mood Detection**: Identifies joy, sadness, boredom, hunger, and aggressivity
- **Behavioral Recommendations**: Provides actionable suggestions for each mood

### 🖼️ **Skin Condition Analysis** 
- **Image Classification**: Analyzes photos of dog skin conditions
- **Medical Detection**: Identifies ear infections, atopic dermatitis, acute dermatitis, and lick granulomas
- **Veterinary Guidance**: Provides care recommendations and when to see a vet

### 🤖 **AI-Powered Suggestions**
- **LLM Integration**: Uses Anthropic Claude for intelligent, contextual recommendations
- **Product Recommendations**: Suggests specific products for each condition
- **Safety-First**: Emphasizes when professional veterinary care is needed

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- FFmpeg (for audio processing)
- Virtual environment


## 📡 API Endpoints

### Health Check
```bash
GET /health
```
Returns system status and available classifiers.

### Unified Analysis
```bash
POST /analyze
Content-Type: multipart/form-data
Body: file=<audio_or_image_file>
```

**Supported file types:**
- **Audio**: `.wav` (for now)
- **Images**: `.png` (for now)

## 🤖 Machine Learning Models

### 🔊 **Bark Classifier (K-Nearest Neighbors)**
- **Algorithm**: K-Nearest Neighbors (KNN) with k=5
- **Feature Extraction**: Spectral audio features including:
  - Spectral centroid (brightness of sound)
  - Spectral roll-off (frequency distribution shape)
  - Zero-crossing rate (speech vs music characteristics)
  - Spectral flatness (noisiness measure)
  - Dominant frequency and magnitude
  - RMS energy (loudness measure)
- **Training Data**: 5 labeled audio files (1.wav - 5.wav) segmented into multiple samples
- **Labels**: joy, boredom, hunger, aggressivity, sadness
- **Audio Processing**: FFmpeg converts input to mono 16kHz WAV format
- **Segmentation**: Audio split into overlapping windows for robust classification

### 🖼️ **Skin Condition Classifier (K-Nearest Neighbors)**
- **Algorithm**: K-Nearest Neighbors with scikit-learn pipeline
- **Feature Extraction**: Computer vision features including:
  - Local Binary Pattern (LBP) for texture analysis
  - Color histogram features
  - Spatial texture descriptors
- **Training Data**: Organized image datasets by condition type:
  - `ear/` - Ear infection samples
  - `atopic/` - Atopic dermatitis samples  
  - `acute/` - Acute moist dermatitis samples
  - `lick/` - Lick dermatitis samples
- **Labels**: ear, atopic, acute, lick
- **Image Processing**: PIL-based preprocessing (resize to 256x256, RGB normalization)
- **Model Storage**: Pre-trained pipeline saved as `knn_skin.joblib`

### 🧠 **AI Suggestion Engine**
- **Primary**: Anthropic Claude 3.5 Haiku for contextual recommendations
- **Fallback**: Rule-based system for reliability
- **Features**:
  - Condition-specific prompts (veterinary vs behavioral)
  - Product recommendations tailored to detected conditions
  - Safety-first approach emphasizing professional care when needed
  - JSON-structured responses with state, suggestion, products, and reasoning

