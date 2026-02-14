# NeuralRot Guide

## Walton's guide
```
To train: 
python main.py all --labels dab,neutral,thumbs_up,six_seven,peace,ok,middle_finger --samples-per-label 100

To run:
python main.py app
```
## What This Project Does
NeuralRot classifies body gestures from webcam video and shows predictions in a frontend UI.

Current pipeline:
1. Collect gesture samples (`gesture_data.csv`)
2. Train a classifier (`gesture_model.pkl`)
3. Run backend API (`/predict`)
4. Run frontend UI (webcam + live prediction display)

---

## Project Structure
```text
NeuralRot/
|-- main.py                 # Data collection + training + one-command app launcher
|-- gesture_classifier.py   # Standalone training script with augmentation
|-- backend_api.py          # Flask API for live prediction from frontend frames
|-- meme_engine.py          # Local OpenCV prediction loop (desktop mode)
|-- gesture_data.csv        # Collected training data (label + features)
|-- gesture_model.pkl       # Trained ML model
|-- requirements.txt        # Cross-version dependency ranges
|-- frontend/
|   |-- index.html          # UI
|   |-- script.js           # Webcam + calls backend /predict
|   |-- styles.css          # Styling
|   `-- README.md
`-- memes/                  # Optional media assets
```

---

## Setup
Create/activate a virtual environment, then install dependencies:

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Recommended Python: `3.10` to `3.12`.

---

## Run Modes

### 1) One-command full app (frontend + backend)
```powershell
python main.py --run-app
```

Then open:
- `http://127.0.0.1:5500`

Backend health:
- `http://127.0.0.1:8000/health`

### 2) Collect data + train model
```powershell
python main.py all --labels dab,neutral --samples-per-label 100
```

This records samples and trains `gesture_model.pkl`.

### 3) Train only from existing CSV
```powershell
python main.py train
```

or with augmentation options:
```powershell
python gesture_classifier.py --csv gesture_data.csv --model gesture_model.pkl --augment-factor 3
```

### 4) Desktop prediction loop (no web frontend)
```powershell
python meme_engine.py
```

---

## Intuitive Commands
```powershell
python main.py collect --labels dab,neutral --samples-per-label 100
python main.py train
python main.py all --labels dab,neutral
python main.py app
```

## End-to-End Flow
1. Frontend captures webcam frame.
2. Frontend sends JPEG base64 to `POST /predict`.
3. `backend_api.py` extracts pose + hands landmarks.
4. Feature vector is normalized and passed to model.
5. Predicted label is returned to frontend.
6. Frontend displays current gesture.

---

## Common Issues

### Frontend shows backend error/offline
- Make sure `python main.py --run-app` is running.
- Verify backend: `http://127.0.0.1:8000/health`.

### No predictions / model not found
- Train first:
```powershell
python main.py train
```
- Confirm `gesture_model.pkl` exists in project root.

### Webcam works but labels are poor
- Collect more balanced data per class.
- Include a `neutral` class.
- Retrain with augmentation:
```powershell
python gesture_classifier.py --augment-factor 5 --jitter-std 0.02 --scale-jitter 0.08
```

