import cv2
import mediapipe as mp
import pandas as pd
import os
import math

GESTURE_LABEL = "dab"   # change per session
CSV_PATH = "gesture_data.csv"

mp_holistic = mp.solutions.holistic
mp_draw = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    enable_segmentation=False,
    refine_face_landmarks=False,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
data = []

print("Press [S] to save sample | [Q] to quit")
print("Collect dab with full body visible (head + shoulders + elbows + wrists)")


def append_landmarks(features, landmarks, anchor, scale, expected_count, use_visibility=False):
    """Append normalized xyz (and optional visibility); fill zeros if missing."""
    if landmarks:
        for lm in landmarks.landmark:
            features.extend([
                (lm.x - anchor[0]) / scale,
                (lm.y - anchor[1]) / scale,
                (lm.z - anchor[2]) / scale
            ])
            if use_visibility:
                features.append(lm.visibility)
    else:
        values_per_point = 4 if use_visibility else 3
        features.extend([0.0] * (expected_count * values_per_point))


def extract_features(results):
    pose = results.pose_landmarks
    left_hand = results.left_hand_landmarks
    right_hand = results.right_hand_landmarks

    if not pose:
        return None

    left_shoulder = pose.landmark[11]
    right_shoulder = pose.landmark[12]

    anchor = (
        (left_shoulder.x + right_shoulder.x) / 2.0,
        (left_shoulder.y + right_shoulder.y) / 2.0,
        (left_shoulder.z + right_shoulder.z) / 2.0
    )

    shoulder_dist = math.sqrt(
        (left_shoulder.x - right_shoulder.x) ** 2 +
        (left_shoulder.y - right_shoulder.y) ** 2 +
        (left_shoulder.z - right_shoulder.z) ** 2
    )
    scale = max(shoulder_dist, 1e-6)

    features = []
    append_landmarks(features, pose, anchor, scale, expected_count=33, use_visibility=True)
    append_landmarks(features, left_hand, anchor, scale, expected_count=21)
    append_landmarks(features, right_hand, anchor, scale, expected_count=21)
    return features


while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)
    landmarks = extract_features(results)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    if results.face_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.face_landmarks,
            mp_holistic.FACEMESH_CONTOURS
        )

    if results.left_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        mp_draw.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    cv2.putText(frame, f"Gesture: {GESTURE_LABEL}", (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    pose_ready = "YES" if results.pose_landmarks else "NO"
    cv2.putText(frame, f"Pose detected: {pose_ready}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)

    cv2.imshow("Holistic Tracking", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s") and landmarks is not None:
        data.append([GESTURE_LABEL] + landmarks)
        print("Saved sample")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
holistic.close()

if data:
    df = pd.DataFrame(data)
    if os.path.exists(CSV_PATH):
        df.to_csv(CSV_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(CSV_PATH, index=False)

    print(f"Saved {len(data)} samples -> {CSV_PATH}")
