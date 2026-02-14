import base64
import math
import os
from io import BytesIO

import cv2
import joblib
import mediapipe as mp
import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

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


def append_landmarks(features, landmarks, anchor, scale, expected_count, use_visibility=False):
    if landmarks:
        for lm in landmarks.landmark:
            features.extend(
                [
                    (lm.x - anchor[0]) / scale,
                    (lm.y - anchor[1]) / scale,
                    (lm.z - anchor[2]) / scale,
                ]
            )
            if use_visibility:
                features.append(lm.visibility)
    else:
        values_per_point = 4 if use_visibility else 3
        features.extend([0.0] * (expected_count * values_per_point))


def extract_features(pose_landmarks, left_hand_landmarks, right_hand_landmarks):
    if not pose_landmarks:
        return None

    left_shoulder = pose_landmarks.landmark[11]
    right_shoulder = pose_landmarks.landmark[12]

    anchor = (
        (left_shoulder.x + right_shoulder.x) / 2.0,
        (left_shoulder.y + right_shoulder.y) / 2.0,
        (left_shoulder.z + right_shoulder.z) / 2.0,
    )

    shoulder_dist = math.sqrt(
        (left_shoulder.x - right_shoulder.x) ** 2
        + (left_shoulder.y - right_shoulder.y) ** 2
        + (left_shoulder.z - right_shoulder.z) ** 2
    )
    scale = max(shoulder_dist, 1e-6)

    features = []
    append_landmarks(features, pose_landmarks, anchor, scale, expected_count=33, use_visibility=True)
    append_landmarks(features, left_hand_landmarks, anchor, scale, expected_count=21)
    append_landmarks(features, right_hand_landmarks, anchor, scale, expected_count=21)
    return features


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
