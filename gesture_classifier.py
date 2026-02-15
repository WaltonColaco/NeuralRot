import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import cv2
import mediapipe as mp

from features import extract_features

CSV_PATH = "gesture_data.csv"
MODEL_PATH = "gesture_model.pkl"


def augment_samples(
    X,
    y,
    factor=2,
    jitter_std=0.01,
    scale_jitter=0.05,
    translate_std=0.01,
    feature_dropout=0.0,
    random_state=42,
):
    """Create synthetic samples from existing rows for low-data training."""
    if factor <= 0:
        return X, y

    rng = np.random.default_rng(random_state)
    X_aug = [X]
    y_aug = [y]

    base = X.values.astype(np.float32)
    labels = y.values

    for _ in range(factor):
        noise = rng.normal(0.0, jitter_std, size=base.shape).astype(np.float32)
        scales = (1.0 + rng.uniform(-scale_jitter, scale_jitter, size=(base.shape[0], 1))).astype(np.float32)
        shifts = rng.normal(0.0, translate_std, size=(base.shape[0], 1)).astype(np.float32)

        synthetic = base * scales + noise + shifts

        if feature_dropout > 0.0:
            drop_mask = rng.random(size=base.shape) < feature_dropout
            synthetic[drop_mask] = 0.0

        X_aug.append(pd.DataFrame(synthetic, columns=X.columns))
        y_aug.append(pd.Series(labels, name=y.name))

    return pd.concat(X_aug, ignore_index=True), pd.concat(y_aug, ignore_index=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train gesture classifier with optional augmentation.")
    parser.add_argument("--csv", default=CSV_PATH, help="Input dataset CSV")
    parser.add_argument("--model", default=MODEL_PATH, help="Output model path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--augment-factor", type=int, default=2, help="Synthetic copies per original row")
    parser.add_argument("--jitter-std", type=float, default=0.01, help="Gaussian noise std")
    parser.add_argument("--scale-jitter", type=float, default=0.05, help="Per-sample scale jitter range")
    parser.add_argument("--translate-std", type=float, default=0.01, help="Per-sample translation std")
    parser.add_argument("--feature-dropout", type=float, default=0.0, help="Probability of zeroing features")
    parser.add_argument("--estimators", type=int, default=300, help="Number of trees")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--live", action="store_true", help="Launch real-time classification after training")
    return parser.parse_args()


def run_live(model):
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

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

    print("Perform an action to classify. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        if pose_results.pose_landmarks:
            mp_draw.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        if left_hand_landmarks:
            mp_draw.draw_landmarks(frame, left_hand_landmarks, mp_hands.HAND_CONNECTIONS)
        if right_hand_landmarks:
            mp_draw.draw_landmarks(frame, right_hand_landmarks, mp_hands.HAND_CONNECTIONS)

        label = "No pose"
        features = extract_features(
            pose_results.pose_landmarks,
            left_hand_landmarks,
            right_hand_landmarks,
        )

        if features is not None:
            label = model.predict([features])[0]
            print(f"Action classified as: {label}")

        cv2.putText(frame, f"Gesture: {label}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Action Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()


def main():
    args = parse_args()

    df = pd.read_csv(args.csv)
    if df.empty:
        raise ValueError("CSV is empty. Collect data first.")

    X = df.iloc[:, 1:]
    y = df.iloc[:, 0]

    if y.nunique() < 2:
        raise ValueError("Need at least 2 gesture labels to train.")

    X_aug, y_aug = augment_samples(
        X,
        y,
        factor=args.augment_factor,
        jitter_std=args.jitter_std,
        scale_jitter=args.scale_jitter,
        translate_std=args.translate_std,
        feature_dropout=args.feature_dropout,
        random_state=args.seed,
    )

    class_counts = y_aug.value_counts()
    use_stratify = class_counts.min() >= 2

    X_train, X_test, y_train, y_test = train_test_split(
        X_aug,
        y_aug,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=y_aug if use_stratify else None,
    )

    model = RandomForestClassifier(
        n_estimators=args.estimators,
        max_depth=None,
        random_state=args.seed,
        class_weight="balanced_subsample",
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"Original rows: {len(X)}")
    print(f"Rows after augmentation: {len(X_aug)}")
    print(f"Classes: {sorted(y_aug.unique())}")
    print(f"Accuracy: {acc:.4f}")

    joblib.dump(model, args.model)
    print(f"Model saved: {args.model}")

    if args.live:
        run_live(model)


if __name__ == "__main__":
    main()
