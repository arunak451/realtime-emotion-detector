import cv2
import numpy as np
import time
from deepface import DeepFace

class SimpleEmotionDetector:
    def __init__(self):
        """
        Simple Emotion Detector using only DeepFace (no MediaPipe)
        """
        print("Initializing Simple Emotion Detector...")
        
        # Load OpenCV's face detector (Haar Cascade)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Emotion labels and colors
        self.emotion_colors = {
            'angry': (0, 0, 255),        # Red
            'disgust': (0, 128, 0),      # Green
            'fear': (128, 0, 128),       # Purple
            'happy': (0, 255, 255),      # Yellow
            'sad': (255, 0, 0),          # Blue
            'surprise': (0, 165, 255),   # Orange
            'neutral': (200, 200, 200)   # Gray
        }
        
        print("Detector ready!")
    
    def detect_faces_opencv(self, frame):
        """Detect faces using OpenCV Haar Cascade"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert format to (x1, y1, x2, y2)
        face_boxes = []
        for (x, y, w, h) in faces:
            x1, y1 = x, y
            x2, y2 = x + w, y + h
            face_boxes.append((x1, y1, x2, y2))
        
        return face_boxes
    
    def run_webcam(self, camera_id=0):
        """Run emotion detection"""
        cap = cv2.VideoCapture(camera_id)
        
        print("\nControls: q=quit, s=save, p=pause")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.detect_faces_opencv(frame)
            
            for (x1, y1, x2, y2) in faces:
                face_img = frame[y1:y2, x1:x2]
                
                try:
                    # Analyze emotion
                    analysis = DeepFace.analyze(
                        face_img,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    
                    emotion = analysis['dominant_emotion']
                    color = self.emotion_colors.get(emotion, (255, 255, 255))
                    
                    # Draw box and label
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, emotion.upper(),
                              (x1, y1-10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                except:
                    pass
            
            cv2.imshow('Emotion Detection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                cv2.imwrite(f'screenshot_{time.time()}.jpg', frame)
        
        cap.release()
        cv2.destroyAllWindows()

# Quick test
if __name__ == "__main__":
    print("Testing Simple Emotion Detector...")
    detector = SimpleEmotionDetector()
    detector.run_webcam()