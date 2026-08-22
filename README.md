
# 🖐️ HandTracker 3D AI

> Real-time hand tracking and gesture-controlled 3D visualization using OpenCV, MediaPipe, NumPy, and Streamlit WebRTC.

HandTracker 3D AI is an interactive computer vision application that detects human hands through a webcam and uses hand gestures to control a real-time 3D wireframe object.

The application combines **MediaPipe Hand Landmarker** for hand detection, **OpenCV** for image processing and visualization, **NumPy** for mathematical calculations, and **Streamlit WebRTC** for live webcam streaming.

---

## ✨ Features

- 🖐️ Real-time hand detection
- 👥 Support for up to 2 hands
- 🎯 21 hand landmarks per hand
- 📐 Gesture-based 3D object control
- 🔍 Dynamic object scaling
- 📍 Hand-controlled object positioning
- 🔄 Hand-controlled rotation
- 🤲 Two-hand gesture support
- 🔁 Automatic rotation when no hand is detected
- 📊 Real-time FPS display
- 🎛️ Interactive Streamlit controls
- 🧊 Multiple 3D wireframe shapes
- 🌑 Modern dark UI
- ⚡ Real-time webcam processing

---

# 🎥 How It Works

The application follows this pipeline:

```text
Webcam
   │
   ▼
Streamlit WebRTC
   │
   ▼
OpenCV Frame Processing
   │
   ▼
MediaPipe Hand Landmarker
   │
   ▼
21 Hand Landmarks
   │
   ▼
Gesture Analysis
   │
   ├── Hand Position
   ├── Thumb–Index Distance
   ├── Hand Angle
   └── Two-Hand Distance
   │
   ▼
3D Transformation
   │
   ├── Position
   ├── Scale
   └── Rotation
   │
   ▼
OpenCV Rendering
   │
   ▼
Live Video Output
```
