import cv2

# Camera Settings
CAMERA_ID = 0  # Default webcam
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FPS = 30

# Detection Settings
YOLO_MODEL_PATH = "yolov8n.pt"  # Nano model for speed
CONFIDENCE_THRESHOLD = 0.5
TARGET_CLASSES = [0]  # 0 is 'person' in COCO dataset. Add more IDs as needed (e.g., 41: cup, 39: bottle)

# Temperature Simulation Settings
# These are heuristic values for demo purposes when no thermal camera is available.
# Format: {class_id: (base_temp, variance)}
SIMULATION_MAP = {
    0: (36.6, 1.5),   # Person: ~37°C
    41: (55.0, 5.0),  # Cup: ~55°C (Hot coffee)
    39: (5.0, 2.0),   # Bottle: ~5°C (Cold drink)
    # Default fallback
    "default": (22.0, 1.0) # Room temp
}

# Visualization Settings
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
TEXT_COLOR = (255, 255, 255)
BOX_COLOR = (0, 255, 0)
TEMP_COLOR_HOT = (0, 0, 255)
TEMP_COLOR_COLD = (255, 0, 0)
TEMP_COLOR_NORMAL = (0, 255, 0)

# Calibration Defaults
DEFAULT_EMISSIVITY = 0.95
DEFAULT_OFFSET = 0.0  # Temperature offset in Celsius
