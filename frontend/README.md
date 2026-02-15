# NeuralRot Frontend

This is the frontend for the NeuralRot project, themed around "Brain Rot". It provides a fun and interactive interface for users to perform gestures and see the corresponding meme output in real-time.

## Features

- Real-time webcam feed
- Gesture detection and classification
- Display detected gesture on the screen
- Start/Stop webcam functionality

## How to Run

1. Open `index.html` in your browser.
2. Click the "Start" button to begin the webcam feed.
3. Perform gestures in front of the camera.
4. The detected gesture will be displayed on the screen.
5. Click the "Stop" button to stop the webcam feed.

## Folder Structure

```
frontend/
├── index.html       # Main HTML file
├── styles.css       # Styling for the frontend
├── script.js        # JavaScript for frontend interactivity
└── README.md        # Documentation for the frontend
```

## Notes

- The gesture detection is currently simulated in the frontend. Replace the `detectGestures` function in `script.js` with actual backend integration for real-time gesture detection.
- Ensure the backend is running and accessible for full functionality.

Enjoy the brain rot! 🧠💀
