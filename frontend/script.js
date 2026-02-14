const video = document.getElementById('webcam');
const canvas = document.getElementById('overlay');
const ctx = canvas.getContext('2d');
const gestureOutput = document.getElementById('gesture');
const startButton = document.getElementById('start-button');
const stopButton = document.getElementById('stop-button');

let stream = null;
let isRunning = false;

async function startWebcam() {
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video: true });
        video.srcObject = stream;
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
    }
}

async function detectGestures() {
    if (!isRunning) return;

    // Simulate gesture detection (replace with actual backend integration)
    setTimeout(() => {
        const gestures = ['dab', '67', 'thumbs_up', 'heart', 'No hand'];
        const randomGesture = gestures[Math.floor(Math.random() * gestures.length)];
        gestureOutput.textContent = randomGesture;

        // Add animation to gesture output
        gestureOutput.classList.add('animate-pulse');
        setTimeout(() => gestureOutput.classList.remove('animate-pulse'), 500);

        // Continue detecting gestures
        detectGestures();
    }, 1000);
}

startButton.addEventListener('click', startWebcam);
stopButton.addEventListener('click', stopWebcam);
