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
        self.current_exercise = "squat"
        
        # Good form definitions
        self.good_forms = {
            "squat": {
                "description": "Keep back straight, knees in line with toes, hips below knees.",
                "thresholds": {"down": 90, "up": 160}
            },
            "curl": {
                "description": "Keep elbows pinned to side. Full extension and flexion.",
                "thresholds": {"down": 160, "up": 30}
            },
            "pushup": {
                "description": "Maintain straight plank position. Chest close to floor.",
                "thresholds": {"down": 90, "up": 160}
            }
        }

    def set_exercise(self, exercise_name: str):
        """Sets the current exercise to track."""
        exercise_name = exercise_name.lower()
        if exercise_name in self.good_forms:
            self.current_exercise = exercise_name
            self.counter = 0
            self.stage = None
            self.feedback = f"Selected: {exercise_name.capitalize()}. {self.good_forms[exercise_name]['description']}"
            return True
        return False

    def _analyze_squat(self, landmarks):
        """Analyzes Squat form and counts reps."""
        # Get coordinates
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_KNEE.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        
        # Calculate knee angle
        angle = calculate_angle(hip, knee, ankle)
        
        # Calculate back angle (approximate with shoulder-hip-vertical? using knee for reference might vary)
        # Using Knee-Hip-Shoulder angle for torso lean
        hip_angle = calculate_angle(knee, hip, shoulder)

        state = self.stage
        if angle > self.good_forms["squat"]["thresholds"]["up"]:
            state = "UP"
        if angle < self.good_forms["squat"]["thresholds"]["down"] and state == 'UP':
            state = "DOWN"
            self.counter += 1
            
        # Form Checks
        feedback = "Good Form"
        if hip_angle < 70: # Torso leaning too forward
            feedback = "Keep Chest Up"
        if state == "DOWN" and angle > 100: # Not deep enough when trying to go down, simplified logic
             feedback = "Go Lower"
             
        return angle, state, feedback, knee # Return knee coordinates for text placement

    def _analyze_curl(self, landmarks):
        """Analyzes Bicep Curl form and counts reps."""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        
        angle = calculate_angle(shoulder, elbow, wrist)
        
        state = self.stage
        if angle > self.good_forms["curl"]["thresholds"]["down"]:
            state = "DOWN"
        if angle < self.good_forms["curl"]["thresholds"]["up"] and state == 'DOWN':
            state = "UP"
            self.counter += 1
            
        # Form Checks
        feedback = "Good Form"
        # Check if elbow is moving too much? (Requires previous frames, skipping for simplicity)
        # Check for full ROM
        if state == "UP" and angle > 45: 
            feedback = "Squeeze at top"
        if state == "DOWN" and angle < 140:
            feedback = "Full Extension"

        return angle, state, feedback, elbow

    def _analyze_pushup(self, landmarks):
        """Analyzes Pushup form and counts reps."""
        shoulder = [landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
        elbow = [landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
        wrist = [landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value].y]
        hip = [landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value].y]
        ankle = [landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].x, landmarks[self.mp_pose.PoseLandmark.LEFT_ANKLE.value].y]
        
        elbow_angle = calculate_angle(shoulder, elbow, wrist)
        body_angle = calculate_angle(shoulder, hip, ankle)
        
        state = self.stage
        if elbow_angle > self.good_forms["pushup"]["thresholds"]["up"]:
            state = "UP"
        if elbow_angle < self.good_forms["pushup"]["thresholds"]["down"] and state == 'UP':
            state = "DOWN"
            self.counter += 1
            
        # Form Checks
        feedback = "Good Form"
        if body_angle < 160 or body_angle > 200: # Simple plank check
            feedback = "Straighten Body"
            
        return elbow_angle, state, feedback, elbow
    

    def _analyze_dynamic(self, landmarks):
        """AI'dan gelen kurallara göre her türlü hareketi analiz eder."""
        try:
            config = self.good_forms[self.current_exercise]
            
            # AI'nın belirlediği eklemleri seç (Örn: Lunge için diz, Pushup için dirsek)
            p1 = landmarks[getattr(self.mp_pose.PoseLandmark, config['landmarks'][0]).value]
            p2 = landmarks[getattr(self.mp_pose.PoseLandmark, config['landmarks'][1]).value]
            p3 = landmarks[getattr(self.mp_pose.PoseLandmark, config['landmarks'][2]).value]

            angle = calculate_angle([p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y])

            # AI'dan gelen eşiklere göre tekrar say
            if angle > config["thresholds"]["up"]:
                self.stage = "UP"
            if angle < config["thresholds"]["down"] and self.stage == 'UP':
                self.stage = "DOWN"
                self.counter += 1
                
            return angle, self.stage, self.feedback, [p2.x, p2.y]
        except Exception as e:
            # Fallback for configuration errors
            print(f"Dynamic Analysis Error: {e}")
            return 0, self.stage, "Cfg Error", [0.5, 0.5]

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
        text_coord = (50, 50) # Default
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # Route logic based on selected exercise
            if self.current_exercise == "squat":
                angle, self.stage, self.feedback, coord_norm = self._analyze_squat(landmarks)
            elif self.current_exercise == "curl":
                angle, self.stage, self.feedback, coord_norm = self._analyze_curl(landmarks)
            elif self.current_exercise == "pushup":
                angle, self.stage, self.feedback, coord_norm = self._analyze_pushup(landmarks)
            else:
                 # Dynamic fallback
                 angle, self.stage, self.feedback, coord_norm = self._analyze_dynamic(landmarks)

            text_coord = tuple(np.multiply(coord_norm, [640, 480]).astype(int))
            
            # Visualize angle
            cv2.putText(image, str(int(angle)), 
                           text_coord, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                                )
            
        except Exception as e:
            pass
        
        # Draw Status Box (Optional UI enhancement)
        cv2.rectangle(image, (0,0), (225,73), (245,117,16), -1)
        
        # Rep Data
        cv2.putText(image, 'REPS', (15,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.counter), (10,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
        
        # Stage Data
        cv2.putText(image, 'STAGE', (65,12), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
        cv2.putText(image, str(self.stage), (60,60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)

        # Feedback Data (New)
        cv2.rectangle(image, (0, 440), (640, 480), (245, 117, 16), -1) 
        cv2.putText(image, f"Mode: {self.current_exercise.upper()} | {self.feedback}", (10,465), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2, cv2.LINE_AA)
        
        # Render detections
        self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        
        return image, {"angle": angle, "state": self.stage, "reps": self.counter, "feedback": self.feedback}
    