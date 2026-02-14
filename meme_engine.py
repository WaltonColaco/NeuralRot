import argparse
import os
import time

import cv2
import imageio
import joblib
import mediapipe as mp
import pygame

from features import extract_features

MODEL_PATH = "gesture_model.pkl"

MEME_MAP = {
    "dab": {"gif": "memes/dab.gif"},
    "six_seven": {"audio": "memes/vine_boom.mp3"},
    "thumbs_up": {"audio": "memes/gigachad.mp3"},
}


def play_audio(path):
    if not os.path.exists(path):
        print(f"Missing audio file: {path}")
        return
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()


def play_gif(path):
    if not os.path.exists(path):
        print(f"Missing gif file: {path}")
        return
    frames = imageio.mimread(path)
    for f in frames:
        img = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        cv2.imshow("MEME", img)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("MEME")


def main():
    parser = argparse.ArgumentParser(description="Desktop gesture prediction with optional meme triggers.")
    parser.add_argument("--model", default=MODEL_PATH, help="Model path")
    parser.add_argument("--memes", action="store_true", help="Enable meme triggers (audio/gif)")
    args = parser.parse_args()

    model = joblib.load(args.model)

    if args.memes:
        pygame.mixer.init()

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

    last_trigger = 0
    cooldown = 2.0
    last_printed_label = None

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
            if label != last_printed_label:
                print(f"Prediction: {label}")
                last_printed_label = label

            if args.memes and label in MEME_MAP and time.time() - last_trigger > cooldown:
                meme = MEME_MAP[label]

                if "audio" in meme:
                    play_audio(meme["audio"])

                if "gif" in meme:
                    play_gif(meme["gif"])

                last_trigger = time.time()

        cv2.putText(frame, f"Gesture: {label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Q to quit", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Meme Engine", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pose.close()
    hands.close()


if __name__ == "__main__":
    main()
