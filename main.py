import cv2
import time
import argparse
from src.video_stream import VideoStream
from src.detector import ObjectDetector
from src.temperature import SimulatedTemperatureEstimator, RegressionTemperatureEstimator
from src.config import *

def main():
    parser = argparse.ArgumentParser(description="Thermal Vision Temperature Estimator")
    parser.add_argument("--source", default=CAMERA_ID, help="Video source (0 for webcam, or path to file)")
    parser.add_argument("--mode", choices=["simulated", "regression"], default="simulated", help="Temperature estimation mode")
    args = parser.parse_args()

    # Handle source argument (int if digit, else string)
    source = int(args.source) if str(args.source).isdigit() else args.source

    # Initialize components
    try:
        stream = VideoStream(source)
    except ValueError as e:
        print(f"Error: {e}")
        return

    detector = ObjectDetector()
    
    if args.mode == "simulated":
        estimator = SimulatedTemperatureEstimator()
        print("Mode: Simulated Temperature (Heuristic)")
    else:
        estimator = RegressionTemperatureEstimator()
        print("Mode: Regression Model (Placeholder)")

    # Window setup
    window_name = "Thermal Vision AI"
    cv2.namedWindow(window_name)

    # Trackbar callback (does nothing, just required by OpenCV)
    def nothing(x):
        pass

    # Create Trackbars
    cv2.createTrackbar("Calibration Offset", window_name, 50, 100, nothing) # 0-100 maps to -50 to +50
    cv2.createTrackbar("Emissivity (%)", window_name, 95, 100, nothing)

    prev_time = 0

    print("Starting video loop. Press 'q' to exit.")

    while True:
        ret, frame = stream.read()
        if not ret:
            print("End of stream.")
            break

        # Read Trackbars
        offset_val = cv2.getTrackbarPos("Calibration Offset", window_name)
        emissivity_val = cv2.getTrackbarPos("Emissivity (%)", window_name)

        # Map offset: 50 is 0. 0 is -50. 100 is +50.
        real_offset = offset_val - 50
        estimator.set_calibration(real_offset)
        estimator.emissivity = emissivity_val / 100.0

        # Detection
        detections = detector.detect(frame)

        # Process detections
        for (x1, y1, x2, y2, cls_id, conf) in detections:
            # Estimate Temperature
            temp = estimator.estimate(frame, (x1, y1, x2, y2), cls_id)
            
            # Color based on temp
            if temp > 50:
                color = TEMP_COLOR_HOT
            elif temp < 10:
                color = TEMP_COLOR_COLD
            else:
                color = TEMP_COLOR_NORMAL

            # Draw Box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw Label
            label = f"{detector.get_class_name(cls_id)}: {temp}C"
            cv2.putText(frame, label, (x1, y1 - 10), FONT, FONT_SCALE, color, 2)

        # FPS Calculation
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time else 0
        prev_time = curr_time

        # HUD
        cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), FONT, 0.7, TEXT_COLOR, 2)
        cv2.putText(frame, f"Mode: {args.mode.upper()}", (10, 60), FONT, 0.7, TEXT_COLOR, 2)
        cv2.putText(frame, f"Offset: {real_offset}C", (10, 90), FONT, 0.7, TEXT_COLOR, 2)
        cv2.putText(frame, f"Emissivity: {estimator.emissivity:.2f}", (10, 120), FONT, 0.7, TEXT_COLOR, 2)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stream.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
