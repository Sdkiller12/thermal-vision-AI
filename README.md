# Thermal Vision AI

A Python-based Computer Vision application that estimates the temperature of detected objects using YOLOv8 and Heuristic/Regression models.

## Features
- **Object Detection**: Uses YOLOv8 (Nano) for real-time detection.
- **Temperature Estimation**:
  - **Simulated Mode**: Estimates temperature based on object class (e.g., Person ~37°C) with realistic noise.
  - **Regression Mode**: Placeholder for a trained RGB-to-Thermal regression model.
- **Real-time UI**: Displays bounding boxes, temperature labels, and a HUD.
- **Calibration**: Trackbars to adjust temperature offset and emissivity in real-time.

## Installation

1. **Prerequisites**: Python 3.8+
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   *Note: This will install `opencv-python`, `ultralytics`, `torch`, etc.*

## Usage

### Run with Webcam (Simulated Mode)
```bash
python main.py
```

### Run with Video File
```bash
python main.py --source path/to/video.mp4
```

### Run in Regression Mode
```bash
python main.py --mode regression
```

### Controls
- **Trackbars**:
  - `Calibration Offset`: Adjust the base temperature offset (+/- 50°C).
  - `Emissivity`: Adjust the emissivity factor (0.0 - 1.0).
- **Keyboard**:
  - Press `q` to quit.

## Project Structure
- `src/`: Source code.
  - `detector.py`: YOLOv8 wrapper.
  - `temperature.py`: Temperature estimation logic.
  - `video_stream.py`: Video capture handler.
  - `config.py`: Configuration settings.
- `models/`: Directory for model weights.
- `data/`: Directory for datasets/videos.
- `train_temperature_model.py`: Skeleton script for training a custom regression model.

## Training a Custom Model
If you have a dataset of RGB images and corresponding temperature values, you can use `train_temperature_model.py` as a starting point to train a PyTorch regression model.

##downloa link for data files
link for download datafiles : https://adas-dataset-v2.flirconservator.com/#downloadguide  
use 7zip to extract the datazip files 
and copy in data folder in the project 
