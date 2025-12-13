import cv2
try:
    import mediapipe as mp
except ImportError:
    mp = None
import numpy as np
from .utils import calculate_angle

class MotionTracker:
    def __init__(self):
        if mp is None:
            raise ImportError("MediaPipe not installed")
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.counter = 0
        self.stage = None
        self.feedback = "Ready"

    def process_frame(self, frame):
        """
        Processes a video frame to detect pose and analyze motion.
        """
        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # Make detection
        results = self.pose.process(image)
    
        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        angle = 0
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Get coordinates for Squat Analysis (Hip, Knee, Ankle)
            hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
            knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
            ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
            
            # Calculate angle
            angle = calculate_angle(hip, knee, ankle)
            
            # Visualize angle
            cv2.putText(image, str(int(angle)), 
                           tuple(np.multiply(knee, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
            # Squat Logic
            if angle > 170:
                self.stage = "UP"
            if angle < 90 and self.stage == 'UP':
                self.stage = "DOWN"
                self.counter += 1
            
        except Exception as e:
            pass
        
        # Render detections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return image, {"angle": angle, "state": self.stage, "reps": self.counter}
