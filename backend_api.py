import base64
import os
from io import BytesIO

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

from features import extract_features

MODEL_PATH = os.environ.get("MODEL_PATH", "gesture_model.pkl")

app = Flask(__name__)
CORS(app)

model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands

pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)


def decode_image(data_url):
    if "," in data_url:
        _, encoded = data_url.split(",", 1)
    else:
        encoded = data_url
    raw = base64.b64decode(encoded)
    image = Image.open(BytesIO(raw)).convert("RGB")
    rgb = np.array(image)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


@app.get("/health")
def health():
    return jsonify({"ok": True, "model_loaded": model is not None})


@app.get("/")
def root():
    return jsonify(
        {
            "message": "NeuralRot backend is running",
            "frontend_url": "http://127.0.0.1:5500",
            "health_url": "/health",
            "predict_url": "/predict",
        }
    )


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": f"Model not found. Train first: {MODEL_PATH}"}), 503

    payload = request.get_json(silent=True) or {}
    if "image" not in payload:
        return jsonify({"error": "Missing 'image' in request body"}), 400

    try:
        frame = decode_image(payload["image"])
    except Exception:
        return jsonify({"error": "Invalid image payload"}), 400

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pose_results = pose.process(rgb)
    hand_results = hands.process(rgb)

    left_hand_landmarks = None
    right_hand_landmarks = None
    if hand_results.multi_hand_landmarks and hand_results.multi_handedness:
        for hand_lm, handedness in zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness):
            side = handedness.classification[0].label
            if side == "Left":
                left_hand_landmarks = hand_lm
            elif side == "Right":
                right_hand_landmarks = hand_lm

    features = extract_features(
        pose_results.pose_landmarks,
        left_hand_landmarks,
        right_hand_landmarks,
    )

    if features is None:
        return jsonify({"label": "No pose", "has_pose": False})

    label = model.predict([features])[0]
    response = {"label": str(label), "has_pose": True}

    if hasattr(model, "predict_proba"):
        try:
            probs = model.predict_proba([features])[0]
            response["confidence"] = float(np.max(probs))
        except Exception:
            pass

    return jsonify(response)


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=int(os.environ.get("BACKEND_PORT", "8000")),
        debug=False,
    )
