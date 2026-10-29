# 😊 Real-time Emotion Detector

Detects faces from a live webcam feed and classifies each face's emotion in real time, drawing a colour-coded box and label on screen.

## How it works

1. **Face detection:** OpenCV Haar Cascade finds faces in each frame
2. **Emotion recognition:** each face crop is passed to DeepFace's pre-trained deep learning emotion model
3. **Visualisation:** bounding box and label coloured by emotion

Detects 7 emotions: **angry, disgust, fear, happy, sad, surprise, neutral**.

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `s` | Save a screenshot |

## Tech stack

Python · OpenCV · DeepFace · TensorFlow · NumPy

## Run locally

```bash
git clone https://github.com/arunak451/realtime-emotion-detector.git
cd realtime-emotion-detector
pip install opencv-python deepface tf-keras numpy
python emotiondet.py
```

Requires a webcam. The DeepFace model weights download automatically on first run.
