#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera interface for trash classification system.
Handles different camera types and provides a unified interface.
"""

import os
import cv2
import time
import logging
import threading
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List, Dict, Any, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CameraInterface(ABC):
    """Abstract base class for camera interfaces."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the camera."""
        pass
    
    @abstractmethod
    def capture_image(self) -> np.ndarray:
        """Capture a single image."""
        pass
    
    @abstractmethod
    def capture_batch(self, batch_size: int) -> List[np.ndarray]:
        """Capture a batch of images."""
        pass
    
    @abstractmethod
    def release(self) -> None:
        """Release camera resources."""
        pass
    
    @abstractmethod
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information."""
        pass


class OpenCVCamera(CameraInterface):
    """Implementation of camera interface using OpenCV."""
    
    def __init__(self, 
                 camera_id: int = 0, 
                 resolution: Tuple[int, int] = (640, 480),
                 fps: int = 30,
                 auto_exposure: bool = True,
                 exposure_value: int = -5,
                 auto_white_balance: bool = True) -> None:
        """
        Initialize OpenCV camera.
        
        Args:
            camera_id: Camera ID (usually 0 for built-in webcam)
            resolution: Image resolution as (width, height)
            fps: Frames per second
            auto_exposure: Whether to use auto exposure
            exposure_value: Exposure value if auto_exposure is False
            auto_white_balance: Whether to use auto white balance
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps
        self.auto_exposure = auto_exposure
        self.exposure_value = exposure_value
        self.auto_white_balance = auto_white_balance
        self.cap = None
        self.is_initialized = False
        self.lock = threading.Lock()
        
    def initialize(self) -> bool:
        """Initialize the camera with the specified settings."""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera {self.camera_id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            if not self.auto_exposure:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)  # Disable auto exposure
                self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure_value)
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # Enable auto exposure
                
            if not self.auto_white_balance:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 0)  # Disable auto white balance
            else:
                self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)  # Enable auto white balance
            
            # Check if settings were applied correctly
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized with resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            # Warm up the camera
            for _ in range(10):
                ret, _ = self.cap.read()
                if not ret:
                    logger.warning("Failed to read frame during warmup")
                time.sleep(0.05)
                
            self.is_initialized = True
            return True
        
        except Exception as e:
            logger.error(f"Error initializing camera: {str(e)}")
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            return False
    
    def capture_image(self) -> np.ndarray:
        """Capture a single image from the camera."""
        if not self.is_initialized or self.cap is None:
            logger.error("Camera not initialized. Call initialize() first.")
            return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        with self.lock:
            ret, frame = self.cap.read()
            
        if not ret:
            logger.warning("Failed to capture image")
            return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        return frame
    
    def capture_batch(self, batch_size: int) -> List[np.ndarray]:
        """Capture a batch of images."""
        images = []
        for _ in range(batch_size):
            image = self.capture_image()
            images.append(image)
            time.sleep(0.05)  # Small delay between captures
        
        return images
    
    def release(self) -> None:
        """Release camera resources."""
        if self.cap is not None:
            with self.lock:
                self.cap.release()
                self.cap = None
            self.is_initialized = False
            logger.info("Camera resources released")
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information."""
        if not self.is_initialized or self.cap is None:
            return {
                "status": "not_initialized",
                "camera_id": self.camera_id,
                "resolution": self.resolution,
                "fps": self.fps
            }
        
        with self.lock:
            info = {
                "status": "initialized",
                "camera_id": self.camera_id,
                "resolution": (
                    int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ),
                "fps": self.cap.get(cv2.CAP_PROP_FPS),
                "brightness": self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
                "contrast": self.cap.get(cv2.CAP_PROP_CONTRAST),
                "saturation": self.cap.get(cv2.CAP_PROP_SATURATION),
                "hue": self.cap.get(cv2.CAP_PROP_HUE),
                "gain": self.cap.get(cv2.CAP_PROP_GAIN),
                "exposure": self.cap.get(cv2.CAP_PROP_EXPOSURE),
                "auto_exposure": self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            }
        
        return info
    
    def __del__(self):
        """Destructor to ensure camera resources are released."""
        self.release()


class RaspberryPiCamera(CameraInterface):
    """Implementation of camera interface for Raspberry Pi Camera."""
    
    def __init__(self, 
                 resolution: Tuple[int, int] = (640, 480),
                 framerate: int = 30,
                 sensor_mode: int = 0,
                 brightness: int = 50,
                 contrast: int = 0,
                 iso: int = 0) -> None:
        """
        Initialize Raspberry Pi Camera.
        
        Args:
            resolution: Image resolution as (width, height)
            framerate: Frames per second
            sensor_mode: Sensor mode (0 for automatic)
            brightness: Brightness level (0-100)
            contrast: Contrast level (-100 to 100)
            iso: ISO (0 for automatic, or 100, 200, 400, 800)
        """
        self.resolution = resolution
        self.framerate = framerate
        self.sensor_mode = sensor_mode
        self.brightness = brightness
        self.contrast = contrast
        self.iso = iso
        self.camera = None
        self.is_initialized = False
        self.lock = threading.Lock()
    
    def initialize(self) -> bool:
        """Initialize the Raspberry Pi camera."""
        try:
            # Import picamera module here to avoid dependency issues on non-RPi systems
            try:
                from picamera import PiCamera
                from picamera.array import PiRGBArray
            except ImportError:
                logger.error("PiCamera module not found. Is this running on a Raspberry Pi?")
                return False
            
            self.PiCamera = PiCamera
            self.PiRGBArray = PiRGBArray
            
            self.camera = PiCamera()
            self.camera.resolution = self.resolution
            self.camera.framerate = self.framerate
            self.camera.sensor_mode = self.sensor_mode
            self.camera.brightness = self.brightness
            self.camera.contrast = self.contrast
            
            if self.iso > 0:
                self.camera.iso = self.iso
            
            # Allow the camera to warmup
            time.sleep(2)
            
            logger.info(f"Raspberry Pi Camera initialized with resolution: {self.resolution}, FPS: {self.framerate}")
            self.is_initialized = True
            return True
        
        except Exception as e:
            logger.error(f"Error initializing Raspberry Pi Camera: {str(e)}")
            if self.camera is not None:
                self.camera.close()
                self.camera = None
            return False
    
    def capture_image(self) -> np.ndarray:
        """Capture a single image from the Raspberry Pi camera."""
        if not self.is_initialized or self.camera is None:
            logger.error("Camera not initialized. Call initialize() first.")
            return np.zeros((self.resolution[1], self.resolution[0], 3), dtype=np.uint8)
        
        with self.lock:
            # Capture image to a numpy array
            output = self.PiRGBArray(self.camera, size=self.resolution)
            self.camera.capture(output, format='rgb')
            image = output.array
            
        return image
    
    def capture_batch(self, batch_size: int) -> List[np.ndarray]:
        """Capture a batch of images."""
        images = []
        for _ in range(batch_size):
            image = self.capture_image()
            images.append(image)
            time.sleep(0.05)  # Small delay between captures
        
        return images
    
    def release(self) -> None:
        """Release camera resources."""
        if self.camera is not None:
            with self.lock:
                self.camera.close()
                self.camera = None
            self.is_initialized = False
            logger.info("Raspberry Pi Camera resources released")
    
    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information."""
        if not self.is_initialized or self.camera is None:
            return {
                "status": "not_initialized",
                "resolution": self.resolution,
                "framerate": self.framerate
            }
        
        with self.lock:
            info = {
                "status": "initialized",
                "resolution": self.camera.resolution,
                "framerate": self.camera.framerate,
                "sensor_mode": self.camera.sensor_mode,
                "brightness": self.camera.brightness,
                "contrast": self.camera.contrast,
                "iso": self.camera.iso,
                "exposure_mode": self.camera.exposure_mode,
                "awb_mode": self.camera.awb_mode,
            }
        
        return info
    
    def __del__(self):
        """Destructor to ensure camera resources are released."""
        self.release()


class CameraFactory:
    """Factory class to create camera objects based on platform and configuration."""
    
    @staticmethod
    def create_camera(camera_type: str, **kwargs) -> CameraInterface:
        """
        Create a camera object based on the camera type.
        
        Args:
            camera_type: Type of camera ('opencv', 'raspberrypi')
            **kwargs: Additional parameters for the specific camera type
        
        Returns:
            A CameraInterface implementation
            
        Raises:
            ValueError: If the camera type is not supported
        """
        if camera_type.lower() == 'opencv':
            return OpenCVCamera(**kwargs)
        elif camera_type.lower() in ['raspberrypi', 'picamera', 'rpi']:
            return RaspberryPiCamera(**kwargs)
        else:
            raise ValueError(f"Unsupported camera type: {camera_type}")


class CameraManager:
    """Manages camera lifecycle and provides utility functions."""
    
    def __init__(self, camera_type: str, **kwargs):
        """
        Initialize the camera manager.
        
        Args:
            camera_type: Type of camera to use
            **kwargs: Additional parameters for the camera
        """
        self.camera = CameraFactory.create_camera(camera_type, **kwargs)
        self.initialized = False
    
    def setup(self) -> bool:
        """Setup the camera."""
        if self.camera.initialize():
            self.initialized = True
            return True
        return False
    
    def get_frame(self) -> np.ndarray:
        """Get a single frame from the camera."""
        if not self.initialized:
            logger.error("Camera not initialized. Call setup() first.")
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        return self.camera.capture_image()
    
    def get_batch(self, batch_size: int) -> List[np.ndarray]:
        """Get a batch of frames from the camera."""
        if not self.initialized:
            logger.error("Camera not initialized. Call setup() first.")
            empty_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            return [empty_frame] * batch_size
        
        return self.camera.capture_batch(batch_size)
    
    def cleanup(self) -> None:
        """Release camera resources."""
        if self.initialized:
            self.camera.release()
            self.initialized = False
    
    def __enter__(self):
        """Context manager enter method."""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self.cleanup()


# Example usage
if __name__ == "__main__":
    # Example: Using the camera manager with OpenCV
    camera_manager = CameraManager("opencv", camera_id=0, resolution=(1280, 720), fps=30)
    
    try:
        if camera_manager.setup():
            print("Camera initialized successfully")
            
            # Capture a single frame
            frame = camera_manager.get_frame()
            print(f"Frame shape: {frame.shape}")
            
            # Capture a batch of frames
            batch = camera_manager.get_batch(5)
            print(f"Captured batch of {len(batch)} frames")
            
            # Display the first frame
            cv2.imshow("Captured Frame", batch[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Failed to initialize camera")
    
    finally:
        camera_manager.cleanup()
        print("Camera resources released") 