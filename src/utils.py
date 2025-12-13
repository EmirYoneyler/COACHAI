import numpy as np
import cv2

def calculate_angle(a: list, b: list, c: list) -> float:
    """
    Calculates the angle between three points (a, b, c).
    b is the vertex.
    """
    a = np.array(a) # First
    b = np.array(b) # Mid
    c = np.array(c) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
        
    return angle

def draw_landmarks_on_image(image, results, mp_pose, mp_drawing):
    """
    Draws pose landmarks on the image.
    """
    mp_drawing.draw_landmarks(
        image, 
        results.pose_landmarks, 
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
    )
    return image
