#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main application module for trash classification system.
Integrates camera, preprocessing, model inference, and LED feedback.
"""

import os
import sys
import time
import yaml
import argparse
import logging
import threading
import queue
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import modules
from createx.core.camera import CameraManager, CameraInterface
from createx.core.preprocessing import TrashPreprocessingPipeline
from createx.core.inference import ModelFactory, TrashClassifier
from createx.core.led_control import LEDFactory, LEDManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('createx.log')
    ]
)
logger = logging.getLogger(__name__)


class TrashClassificationSystem:
    """Main trash classification system class."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize the trash classification system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.camera_manager = None
        self.pipeline = None
        self.classifier = None
        self.led_manager = None
        
        # Processing queues
        self.frame_queue = queue.Queue(maxsize=30)
        self.result_queue = queue.Queue(maxsize=30)
        
        # Control flags
        self.running = False
        self.initialized = False
        
        # Processing threads
        self.camera_thread = None
        self.processing_thread = None
        self.feedback_thread = None
        
        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'inference_count': 0,
            'avg_inference_time': 0,
            'start_time': 0,
            'avg_fps': 0
        }
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        # Default configuration
        default_config = {
            'camera': {
                'type': 'opencv',
                'device_id': 0,
                'resolution': [640, 480],
                'fps': 30
            },
            'preprocessing': {
                'target_size': [224, 224],
                'detection_method': 'contour',
                'normalize': True,
                'batch_size': 8
            },
            'model': {
                'type': 'pytorch',
                'path': 'models/trash_classifier.pt',
                'device': 'cpu',
                'num_classes': 8,
                'quantized': False,
                'confidence_threshold': 0.7
            },
            'led': {
                'type': 'dummy',
                'red_pin': 17,
                'green_pin': 27,
                'blue_pin': 22,
                'common_anode': True
            },
            'system': {
                'batch_processing': True,
                'batch_size': 8,
                'detection_interval': 1.0,  # seconds
                'feedback_duration': 2.0,  # seconds
                'continuous_operation': True,
                'video_output': False,
                'video_output_path': 'output.mp4'
            }
        }
        
        # Load configuration from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                
                # Merge with default configuration
                if loaded_config:
                    # Update nested dictionaries
                    for key, value in loaded_config.items():
                        if key in default_config and isinstance(value, dict):
                            default_config[key].update(value)
                        else:
                            default_config[key] = value
                
                logger.info(f"Configuration loaded from {config_path}")
            
            except Exception as e:
                logger.error(f"Error loading configuration: {str(e)}")
        
        return default_config
    
    def initialize(self) -> bool:
        """
        Initialize the trash classification system.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Initialize camera
            camera_config = self.config['camera']
            self.camera_manager = CameraManager(
                camera_type=camera_config.get('type', 'opencv'),
                camera_id=camera_config.get('device_id', 0),
                resolution=tuple(camera_config.get('resolution', [640, 480])),
                fps=camera_config.get('fps', 30)
            )
            
            if not self.camera_manager.setup():
                logger.error("Failed to initialize camera")
                return False
            
            # Initialize preprocessing pipeline
            preproc_config = self.config['preprocessing']
            self.pipeline = TrashPreprocessingPipeline(
                target_size=tuple(preproc_config.get('target_size', [224, 224])),
                detection_method=preproc_config.get('detection_method', 'contour'),
                normalize=preproc_config.get('normalize', True),
                batch_size=preproc_config.get('batch_size', 8)
            )
            
            # Initialize model
            model_config = self.config['model']
            model = ModelFactory.create_model(
                model_type=model_config.get('type', 'pytorch'),
                model_path=model_config.get('path', 'models/trash_classifier.pt'),
                device=model_config.get('device', 'cpu'),
                batch_size=model_config.get('batch_size', 8),
                num_classes=model_config.get('num_classes', 8),
                input_size=tuple(preproc_config.get('target_size', [224, 224])),
                quantized=model_config.get('quantized', False)
            )
            
            self.classifier = TrashClassifier(
                model=model,
                batch_size=model_config.get('batch_size', 8),
                confidence_threshold=model_config.get('confidence_threshold', 0.7)
            )
            
            if not self.classifier.initialize():
                logger.error("Failed to initialize classifier")
                self.camera_manager.cleanup()
                return False
            
            # Initialize LED manager
            led_config = self.config['led']
            led = LEDFactory.create_led(
                led_type=led_config.get('type', 'dummy'),
                red_pin=led_config.get('red_pin', 17),
                green_pin=led_config.get('green_pin', 27),
                blue_pin=led_config.get('blue_pin', 22),
                common_anode=led_config.get('common_anode', True)
            )
            
            self.led_manager = LEDManager()
            self.led_manager.add_led('main', led)
            
            if not self.led_manager.initialize():
                logger.warning("Failed to initialize LED manager, continuing without visual feedback")
            
            self.initialized = True
            logger.info("Trash classification system initialized successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing trash classification system: {str(e)}")
            self.cleanup()
            return False
    
    def start(self) -> bool:
        """
        Start the trash classification system.
        
        Returns:
            True if started successfully, False otherwise
        """
        if not self.initialized:
            logger.error("System not initialized. Call initialize() first.")
            return False
        
        if self.running:
            logger.warning("System is already running")
            return True
        
        # Reset statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'inference_count': 0,
            'avg_inference_time': 0,
            'start_time': time.time(),
            'avg_fps': 0
        }
        
        # Start processing threads
        self.running = True
        
        self.camera_thread = threading.Thread(
            target=self._camera_loop,
            daemon=True
        )
        
        self.processing_thread = threading.Thread(
            target=self._processing_loop,
            daemon=True
        )
        
        self.feedback_thread = threading.Thread(
            target=self._feedback_loop,
            daemon=True
        )
        
        self.camera_thread.start()
        self.processing_thread.start()
        self.feedback_thread.start()
        
        logger.info("Trash classification system started")
        return True
    
    def stop(self) -> None:
        """Stop the trash classification system."""
        if not self.running:
            return
        
        # Stop threads
        self.running = False
        
        # Wait for threads to finish
        if self.camera_thread:
            self.camera_thread.join(timeout=2.0)
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
        
        if self.feedback_thread:
            self.feedback_thread.join(timeout=2.0)
        
        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break
        
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break
        
        # Turn off LED
        if self.led_manager:
            self.led_manager.turn_all_off()
        
        logger.info("Trash classification system stopped")
    
    def cleanup(self) -> None:
        """Clean up resources."""
        # Stop processing
        self.stop()
        
        # Clean up components
        if self.camera_manager:
            self.camera_manager.cleanup()
        
        if self.classifier:
            self.classifier.cleanup()
        
        if self.led_manager:
            self.led_manager.cleanup()
        
        self.initialized = False
        logger.info("Trash classification system resources cleaned up")
    
    def _camera_loop(self) -> None:
        """Camera capture loop."""
        system_config = self.config['system']
        continuous = system_config.get('continuous_operation', True)
        interval = system_config.get('detection_interval', 1.0)
        
        last_capture_time = 0
        
        try:
            while self.running:
                # Check if we should capture a new frame based on interval
                current_time = time.time()
                if current_time - last_capture_time < interval and self.stats['frames_captured'] > 0:
                    time.sleep(0.01)  # Short sleep to prevent CPU spikes
                    continue
                
                # Capture frame
                frame = self.camera_manager.get_frame()
                
                # Put frame in queue
                try:
                    self.frame_queue.put(frame, block=False)
                    self.stats['frames_captured'] += 1
                    last_capture_time = current_time
                    
                    # Update FPS
                    elapsed = current_time - self.stats['start_time']
                    if elapsed > 0:
                        self.stats['avg_fps'] = self.stats['frames_captured'] / elapsed
                    
                except queue.Full:
                    # Queue is full, skip this frame
                    pass
                
                # If not in continuous mode and we've captured enough frames, stop
                if not continuous and self.stats['frames_captured'] >= system_config.get('batch_size', 8):
                    self.running = False
                    break
        
        except Exception as e:
            logger.error(f"Error in camera loop: {str(e)}")
            self.running = False
    
    def _processing_loop(self) -> None:
        """Image processing and inference loop."""
        system_config = self.config['system']
        batch_processing = system_config.get('batch_processing', True)
        batch_size = system_config.get('batch_size', 8)
        
        frames_batch = []
        
        try:
            while self.running:
                # Get frame from queue
                try:
                    frame = self.frame_queue.get(timeout=0.5)
                    
                    if batch_processing:
                        # Add frame to batch
                        frames_batch.append(frame)
                        
                        # Process batch when it reaches the desired size
                        if len(frames_batch) >= batch_size:
                            self._process_batch(frames_batch)
                            frames_batch = []
                    else:
                        # Process single frame
                        self._process_single(frame)
                    
                    self.stats['frames_processed'] += 1
                
                except queue.Empty:
                    # If queue is empty and we have frames in batch, process them
                    if batch_processing and frames_batch:
                        self._process_batch(frames_batch)
                        frames_batch = []
        
        except Exception as e:
            logger.error(f"Error in processing loop: {str(e)}")
            self.running = False
    
    def _process_single(self, frame: np.ndarray) -> None:
        """
        Process a single frame.
        
        Args:
            frame: Input frame
        """
        # Start timing
        start_time = time.time()
        
        # Detect and preprocess objects
        processed_objects, detected_objects = self.pipeline.process_single_image(frame)
        
        # If no objects were detected, return
        if not processed_objects:
            return
        
        # Run inference on each object
        results = []
        for obj in processed_objects:
            result = self.classifier.predict_single(obj)
            results.append(result)
        
        # Get the most confident result
        if results:
            best_result = max(results, key=lambda x: x.get('confidence', 0))
            
            # Put result in queue
            try:
                self.result_queue.put(best_result, block=False)
            except queue.Full:
                pass
        
        # Update statistics
        self.stats['inference_count'] += 1
        inference_time = time.time() - start_time
        self.stats['avg_inference_time'] = ((self.stats['avg_inference_time'] * (self.stats['inference_count'] - 1))
                                          + inference_time) / self.stats['inference_count']
    
    def _process_batch(self, frames: List[np.ndarray]) -> None:
        """
        Process a batch of frames.
        
        Args:
            frames: List of input frames
        """
        # Start timing
        start_time = time.time()
        
        # Process each frame individually to detect objects
        all_objects = []
        all_processed = []
        
        for frame in frames:
            processed_objects, detected_objects = self.pipeline.process_single_image(frame)
            
            if processed_objects:
                all_processed.extend(processed_objects)
                all_objects.append(detected_objects)
        
        # If no objects were detected in any frame, return
        if not all_processed:
            return
        
        # Convert to numpy array for batch processing
        batch = np.array(all_processed)
        
        # Run batch inference
        batch_results = self.classifier.predict_batch(batch)
        
        # Average results
        avg_result = self.classifier.batch_average(batch_results)
        
        # Put result in queue
        try:
            self.result_queue.put(avg_result, block=False)
        except queue.Full:
            pass
        
        # Update statistics
        self.stats['inference_count'] += 1
        inference_time = time.time() - start_time
        self.stats['avg_inference_time'] = ((self.stats['avg_inference_time'] * (self.stats['inference_count'] - 1))
                                          + inference_time) / self.stats['inference_count']
    
    def _feedback_loop(self) -> None:
        """LED feedback loop."""
        system_config = self.config['system']
        feedback_duration = system_config.get('feedback_duration', 2.0)
        
        # Turn on white LED to indicate system is ready
        if self.led_manager:
            self.led_manager.set_all_color((255, 255, 255))  # White
            time.sleep(1.0)
            self.led_manager.turn_all_off()
        
        last_result = None
        last_feedback_time = 0
        
        try:
            while self.running:
                # Get result from queue
                try:
                    result = self.result_queue.get(timeout=0.5)
                    last_result = result
                    last_feedback_time = time.time()
                    
                    # Provide LED feedback
                    if self.led_manager and 'category' in result:
                        category = result['category']
                        confidence = result.get('confidence', 0.0)
                        
                        # Log the result
                        logger.info(f"Detected: {category} (confidence: {confidence:.2f})")
                        
                        # Flash LED with category color
                        self.led_manager.flash_category(category)
                
                except queue.Empty:
                    # If no result is available, check if we should turn off LED
                    current_time = time.time()
                    if (last_result is not None and self.led_manager and
                            current_time - last_feedback_time > feedback_duration):
                        self.led_manager.turn_all_off()
                        last_result = None
        
        except Exception as e:
            logger.error(f"Error in feedback loop: {str(e)}")
            self.running = False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get system statistics.
        
        Returns:
            Statistics dictionary
        """
        # Calculate elapsed time
        elapsed = time.time() - self.stats['start_time']
        
        # Update FPS
        if elapsed > 0:
            self.stats['avg_fps'] = self.stats['frames_processed'] / elapsed
        
        return {
            'frames_captured': self.stats['frames_captured'],
            'frames_processed': self.stats['frames_processed'],
            'inference_count': self.stats['inference_count'],
            'avg_inference_time_ms': self.stats['avg_inference_time'] * 1000,
            'avg_fps': self.stats['avg_fps'],
            'running_time_s': elapsed,
            'status': 'running' if self.running else 'stopped'
        }
    
    def __enter__(self):
        """Context manager enter method."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self.cleanup()


def main():
    """Main function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Trash Classification System')
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--model', type=str, help='Path to model file')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to use for inference')
    parser.add_argument('--camera', type=int, help='Camera device ID')
    parser.add_argument('--led-type', type=str, choices=['dummy', 'gpio', 'gpiozero'], help='LED type')
    parser.add_argument('--continuous', action='store_true', help='Run in continuous mode')
    parser.add_argument('--stats', action='store_true', help='Print statistics periodically')
    parser.add_argument('--interval', type=float, help='Detection interval in seconds')
    parser.add_argument('--threshold', type=float, help='Confidence threshold')
    args = parser.parse_args()
    
    # Create configuration based on command line arguments
    config_path = args.config
    config_overrides = {}
    
    if args.model:
        config_overrides.setdefault('model', {})['path'] = args.model
    
    if args.device:
        config_overrides.setdefault('model', {})['device'] = args.device
    
    if args.camera is not None:
        config_overrides.setdefault('camera', {})['device_id'] = args.camera
    
    if args.led_type:
        config_overrides.setdefault('led', {})['type'] = args.led_type
    
    if args.continuous:
        config_overrides.setdefault('system', {})['continuous_operation'] = True
    
    if args.interval:
        config_overrides.setdefault('system', {})['detection_interval'] = args.interval
    
    if args.threshold:
        config_overrides.setdefault('model', {})['confidence_threshold'] = args.threshold
    
    # Create trash classification system
    system = TrashClassificationSystem(config_path)
    
    # Override configuration if needed
    for section, values in config_overrides.items():
        system.config.setdefault(section, {}).update(values)
    
    try:
        # Initialize and start the system
        if system.initialize():
            system.start()
            
            # Main loop
            print("Trash classification system is running. Press Ctrl+C to stop.")
            
            while system.running:
                if args.stats:
                    stats = system.get_stats()
                    print(f"\rFPS: {stats['avg_fps']:.2f}, "
                          f"Inference time: {stats['avg_inference_time_ms']:.2f} ms, "
                          f"Processed: {stats['frames_processed']}", end='')
                
                time.sleep(1.0)
        
        else:
            print("Failed to initialize trash classification system")
    
    except KeyboardInterrupt:
        print("\nStopping trash classification system...")
    
    finally:
        # Clean up resources
        system.cleanup()
        print("Trash classification system stopped")


if __name__ == "__main__":
    main() 