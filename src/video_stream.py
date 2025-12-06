import cv2
from .config import CAMERA_ID, FRAME_WIDTH, FRAME_HEIGHT, FPS

class VideoStream:
    def __init__(self, source=CAMERA_ID):
        """
        Initialize video stream from webcam or file.
        """
        self.cap = cv2.VideoCapture(source)
        
        # Set properties if it's a webcam (source is int)
        if isinstance(source, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FPS)

        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")

    def read(self):
        """
        Read a frame from the stream.
        """
        ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        """
        Release resources.
        """
        self.cap.release()
