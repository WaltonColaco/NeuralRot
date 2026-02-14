import cv2
import mediapipe as mp
import joblib
import pygame
import imageio
import time

MODEL_PATH = "gesture_model.pkl"

MEME_MAP = {
    "dab": {"gif": "memes/dab.gif"},
    "67": {"audio": "memes/vine_boom.mp3"},
    "thumbs_up": {"audio": "memes/gigachad.mp3"},
}

model = joblib.load(MODEL_PATH)

pygame.mixer.init()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

last_trigger = 0
cooldown = 2.0

def play_audio(path):
    pygame.mixer.music.load(path)
    pygame.mixer.music.play()

def play_gif(path):
    frames = imageio.mimread(path)
    for f in frames:
        img = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        cv2.imshow("MEME", img)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break
    cv2.destroyWindow("MEME")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    label = "No hand"

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

        base = hand.landmark[0]
        landmarks = []

        for lm in hand.landmark:
            landmarks.extend([
                lm.x - base.x,
                lm.y - base.y,
                lm.z - base.z
            ])

        label = model.predict([landmarks])[0]

        if label in MEME_MAP and time.time() - last_trigger > cooldown:
            meme = MEME_MAP[label]

            if "audio" in meme:
                play_audio(meme["audio"])

            if "gif" in meme:
                play_gif(meme["gif"])

            last_trigger = time.time()

    cv2.putText(frame, f"Gesture: {label}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Meme Engine", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()