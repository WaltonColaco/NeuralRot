# NeuralRot

NeuralRot classifies webcam gestures using pose + hand landmarks, then serves live predictions to a web frontend.

## Pipeline
1. Data collection:
   - `main.py` captures webcam frames.
   - MediaPipe extracts `pose + left hand + right hand` landmarks.
   - Features are normalized and appended to `gesture_data.csv`.
2. Training:
   - Reads `gesture_data.csv`.
   - Trains a `RandomForestClassifier`.
   - Saves model to `gesture_model.pkl`.
3. Final running:
   - Frontend captures webcam frames.
   - Backend (`/predict`) extracts the same features and predicts label.
   - Frontend displays the predicted gesture live.

## Project Structure
```text
NeuralRot/
|-- main.py
|-- gesture_classifier.py
|-- backend_api.py
|-- meme_engine.py
|-- gesture_data.csv
|-- gesture_model.pkl
|-- requirements.txt
|-- frontend/
|   |-- index.html
|   |-- script.js
|   `-- styles.css
`-- memes/
```

## Setup
```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

Recommended Python: `3.10` to `3.12`.

## Training
### Collect + train in one command
```powershell
python main.py --labels dab,neutral --samples-per-label 100
```

### Train only from existing CSV
```powershell
python main.py --skip-collect
```

### Optional augmented training
```powershell
python gesture_classifier.py --csv gesture_data.csv --model gesture_model.pkl --augment-factor 3
```

## Final Run (Frontend + Backend)
```powershell
python main.py --run-app
```

Open:
- `http://127.0.0.1:5500`

Health check:
- `http://127.0.0.1:8000/health`

## Notes
- If predictions do not work, confirm `gesture_model.pkl` exists.
- If model is missing, train first with `python main.py --skip-collect`.
- Keep head/shoulders/elbows/hands visible during collection and inference.
