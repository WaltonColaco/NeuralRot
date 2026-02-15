const BACKEND_URL = 'http://127.0.0.1:8000';
const PREDICT_INTERVAL_MS = 500;

const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const gestureOutput = document.getElementById('gesture');
const confidenceOutput = document.getElementById('confidence');
const startButton = document.getElementById('start-button');
const stopButton = document.getElementById('stop-button');

let stream = null;
let isRunning = false;

async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
        await video.play();
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        isRunning = true;
        detectGestures();
    } catch (err) {
        console.error('Error accessing webcam:', err);
        alert('Could not access webcam. Please check your permissions.');
    }
}

function stopWebcam() {
    if (stream) {
        const tracks = stream.getTracks();
        tracks.forEach(track => track.stop());
        stream = null;
        isRunning = false;
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        gestureOutput.textContent = 'None';
        confidenceOutput.textContent = '-';
    }
}

function captureFrame() {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.8);
}

async function detectGestures() {
    if (!isRunning) return;

    try {
        const imageData = captureFrame();
        const res = await fetch(`${BACKEND_URL}/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: imageData }),
        });

        if (!res.ok) {
            gestureOutput.textContent = 'Backend error';
        } else {
            const data = await res.json();
            gestureOutput.textContent = (data.label || 'Unknown').replaceAll('_', ' ');
            confidenceOutput.textContent = data.confidence != null
                ? (data.confidence * 100).toFixed(1) + '%'
                : '-';

            gestureOutput.classList.add('animate-bounce');
            setTimeout(() => gestureOutput.classList.remove('animate-bounce'), 500);

            gestureOutput.style.textShadow = '0 0 10px #f6e05e, 0 0 20px #f6e05e';
            setTimeout(() => gestureOutput.style.textShadow = '', 1000);
        }
    } catch (err) {
        console.error('Prediction error:', err);
        gestureOutput.textContent = 'Backend offline';
    }

    setTimeout(detectGestures, PREDICT_INTERVAL_MS);
}

startButton.addEventListener('click', startWebcam);
stopButton.addEventListener('click', stopWebcam);
