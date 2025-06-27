# detection.py
import cv2
from roboflow import Roboflow

def load_detection_model(api_key: str):
    """
    Initialize and return the Roboflow model for football players/ball detection.
    The user should replace project/workspace IDs with their own if needed.
    """
    rf = Roboflow(api_key=api_key)
    # Example: workspace and project from Roboflow Universe
    workspace = rf.workspace("roboflow-jvuqo")
    project = workspace.project("football-players-detection-3zvbc")
    version = project.version(10)  # use version 10 (as example)
    model = version.model
    return model

def detect_players_and_ball(frame, model):
    """
    Perform object detection on the frame using the Roboflow model.
    Returns a list of detections: each is a dict with keys 'class', 'confidence', 'box'=(x,y,w,h).
    """
    # Save frame temporarily for Roboflow (per Roboflow Python SDK usage)
    cv2.imwrite("temp_frame.jpg", frame)
    result = model.predict("temp_frame.jpg", confidence=40, overlap=30).json()
    detections = []
    for pred in result["predictions"]:
        cls = pred["class"]
        confidence = pred["confidence"]
        # roboflow returns center x,y,width,height or x,y,width,height?
        # Assuming result is Pascal VOC style: x,y,width,height
        x = int(pred["x"] - pred["width"]/2)
        y = int(pred["y"] - pred["height"]/2)
        w = int(pred["width"])
        h = int(pred["height"])
        detections.append({
            "class": cls,
            "confidence": confidence,
            "box": (x, y, w, h)
        })
    return detections
