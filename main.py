import argparse
import math
import os
import time

import cv2
import joblib
import mediapipe as mp
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CSV_PATH = "gesture_data.csv"
MODEL_PATH = "gesture_model.pkl"

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils


def append_landmarks(features, landmarks, anchor, scale, expected_count, use_visibility=False):
    if landmarks:
        for lm in landmarks.landmark:
            features.extend([
                (lm.x - anchor[0]) / scale,
                (lm.y - anchor[1]) / scale,
                (lm.z - anchor[2]) / scale,
            ])
            if use_visibility:
                features.append(lm.visibility)
    else:
        values_per_point = 4 if use_visibility else 3
        features.extend([0.0] * (expected_count * values_per_point))


def extract_features(pose_landmarks, left_hand_landmarks, right_hand_landmarks):
    pose = pose_landmarks

    if not pose:
        return None

    left_shoulder = pose.landmark[11]
    right_shoulder = pose.landmark[12]

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
    append_landmarks(features, pose, anchor, scale, expected_count=33, use_visibility=True)
    append_landmarks(features, left_hand_landmarks, anchor, scale, expected_count=21)
    append_landmarks(features, right_hand_landmarks, anchor, scale, expected_count=21)
    return features


def collect_data(labels, samples_per_label, capture_interval, csv_path):
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

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        pose.close()
        hands.close()
        raise RuntimeError("Could not open webcam.")

    print("Press [Q] to stop early.")
    all_rows = []

    try:
        for label in labels:
            collected = 0
            last_capture = 0.0
            print(f"Collecting '{label}' samples...")

            while collected < samples_per_label:
                ret, frame = cap.read()
                if not ret:
                    continue

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

                if pose_results.pose_landmarks:
                    mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
                if left_hand_landmarks:
                    mp_draw.draw_landmarks(frame, left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
                if right_hand_landmarks:
                    mp_draw.draw_landmarks(frame, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

                now = time.time()
                if features is not None and (now - last_capture) >= capture_interval:
                    all_rows.append([label] + features)
                    collected += 1
                    last_capture = now

                cv2.putText(
                    frame,
                    f"Label: {label}  {collected}/{samples_per_label}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "Stand so head + shoulders + elbows are visible",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 200, 255),
                    2,
                )

                cv2.imshow("Auto Dataset Collector", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped early by user.")
                    return write_csv(all_rows, csv_path)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        hands.close()

    return write_csv(all_rows, csv_path)


def write_csv(rows, csv_path):
    if not rows:
        print("No samples captured.")
        return 0

    df = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode="a", header=False, index=False)
    else:
        df.to_csv(csv_path, index=False)

    print(f"Saved {len(rows)} rows -> {csv_path}")
    return len(rows)


def train_model(csv_path, model_path, test_size=0.2):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty. Collect data first.")

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 gesture labels to train a classifier.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=300, max_depth=None, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Accuracy: {acc:.4f}")

    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Collect gesture CSV and train gesture model.")
    parser.add_argument(
        "--labels",
        type=str,
        default="dab,neutral",
        help="Comma-separated gesture labels to collect (e.g. dab,neutral,thumbs_up)",
    )
    parser.add_argument(
        "--samples-per-label",
        type=int,
        default=80,
        help="How many samples to auto-capture per label",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=0.15,
        help="Seconds between captured samples",
    )
    parser.add_argument("--csv", type=str, default=CSV_PATH, help="Dataset CSV path")
    parser.add_argument("--model", type=str, default=MODEL_PATH, help="Model output path")
    parser.add_argument(
        "--skip-collect",
        action="store_true",
        help="Skip webcam capture and train from existing CSV",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    labels = [x.strip() for x in args.labels.split(",") if x.strip()]

    if not args.skip_collect:
        collect_data(
            labels=labels,
            samples_per_label=args.samples_per_label,
            capture_interval=args.capture_interval,
            csv_path=args.csv,
        )

    train_model(csv_path=args.csv, model_path=args.model)


if __name__ == "__main__":
    main()
