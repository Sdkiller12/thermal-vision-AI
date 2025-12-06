import numpy as np
import cv2
from abc import ABC, abstractmethod
from .config import SIMULATION_MAP, DEFAULT_EMISSIVITY, DEFAULT_OFFSET

class TemperatureEstimator(ABC):
    def __init__(self, emissivity=DEFAULT_EMISSIVITY, offset=DEFAULT_OFFSET):
        self.emissivity = emissivity
        self.offset = offset

    @abstractmethod
    def estimate(self, frame, bbox, class_id):
        """
        Estimate temperature for a given bounding box.
        Returns: temperature (float) in Celsius.
        """
        pass

    def set_calibration(self, offset):
        self.offset = offset

class SimulatedTemperatureEstimator(TemperatureEstimator):
    def __init__(self):
        super().__init__()
        self.rng = np.random.default_rng()

    def estimate(self, frame, bbox, class_id):
        """
        Returns a simulated temperature based on the object class.
        Adds some random noise to make it look 'alive'.
        """
        base_temp, variance = SIMULATION_MAP.get(class_id, SIMULATION_MAP["default"])
        
        # Simulate fluctuation
        noise = self.rng.normal(0, 0.2) 
        
        # In a real thermal camera, emissivity affects reading. 
        # Here we just simulate it as a factor (simplified).
        emissivity_factor = 1.0 + (1.0 - self.emissivity) * 0.1
        
        temp = (base_temp + noise) * emissivity_factor + self.offset
        return round(temp, 1)

class RegressionTemperatureEstimator(TemperatureEstimator):
    def __init__(self, model_path=None):
        super().__init__()
        # Placeholder for loading a real trained model
        # self.model = load_model(model_path)
        pass

    def estimate(self, frame, bbox, class_id):
        """
        Predict temperature using pixel data from the bounding box.
        Requires a trained model that maps RGB features to Temperature.
        """
        x1, y1, x2, y2 = bbox
        roi = frame[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.0

        # Placeholder logic: Average pixel intensity mapped to a range
        # This is just a visual placeholder if someone tries to use this mode without a model
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        avg_intensity = np.mean(gray_roi)
        
        # Map 0-255 intensity to 20-40 degrees (arbitrary)
        temp = 20.0 + (avg_intensity / 255.0) * 20.0 + self.offset
        return round(temp, 1)
