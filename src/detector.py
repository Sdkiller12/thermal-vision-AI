from ultralytics import YOLO
import cv2
import numpy as np
from .config import YOLO_MODEL_PATH, CONFIDENCE_THRESHOLD, TARGET_CLASSES

class ObjectDetector:
    def __init__(self, model_path=YOLO_MODEL_PATH):
        """
        Initialize the YOLOv8 detector.
        """
        print(f"Loading YOLO model from {model_path}...")
        self.model = YOLO(model_path)
        self.target_classes = TARGET_CLASSES

    def detect(self, frame):
        """
        Detect objects in the frame.
        Returns a list of detections: [(x1, y1, x2, y2, class_id, confidence), ...]
        """
        results = self.model(frame, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                if conf >= CONFIDENCE_THRESHOLD:
                    # Filter by target classes if specified, otherwise detect all
                    # For this specific project, we might want to detect everything but highlight specific ones
                    # or just filter. Let's filter for now based on config.
                    if self.target_classes is None or cls_id in self.target_classes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        detections.append((x1, y1, x2, y2, cls_id, conf))
        
        return detections

    def get_class_name(self, class_id):
        """
        Return the string name of the class.
        """
        return self.model.names.get(class_id, f"Unknown({class_id})")
