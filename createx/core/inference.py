#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model inference module for trash classification system.
Handles model loading, inference, and result interpretation.
"""

import os
import time
import json
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelInterface(ABC):
    """Abstract base class for model interfaces."""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the model."""
        pass
    
    @abstractmethod
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of preprocessed images.
        
        Args:
            batch: Batch of preprocessed images
            
        Returns:
            Batch of predictions (classification probabilities)
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources."""
        pass


class PyTorchModelInterface(ModelInterface):
    """PyTorch model interface."""
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'cpu',
                 batch_size: int = 8,
                 num_classes: int = 8,
                 input_size: Tuple[int, int] = (224, 224),
                 quantized: bool = False):
        """
        Initialize PyTorch model interface.
        
        Args:
            model_path: Path to the model file
            device: Device to use ('cpu' or 'cuda')
            batch_size: Maximum batch size
            num_classes: Number of classes
            input_size: Input image size
            quantized: Whether the model is quantized
        """
        self.model_path = model_path
        self.device = device
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.quantized = quantized
        
        self.model = None
        self.is_initialized = False
        self.class_names = []
        self.lock = threading.Lock()
    
    def initialize(self) -> bool:
        """Initialize the PyTorch model."""
        try:
            import torch
            from torch import nn
            
            self.torch = torch
            
            # Check if CUDA is available if device is set to 'cuda'
            if self.device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA is not available, falling back to CPU")
                self.device = 'cpu'
            
            # Create torch device
            self.torch_device = torch.device(self.device)
            
            # Load the model
            with self.lock:
                if self.quantized:
                    # Load quantized model
                    self.model = torch.jit.load(self.model_path)
                    self.model.eval()
                    self.model.to(self.torch_device)
                else:
                    # Load regular model
                    self.model = torch.load(self.model_path, map_location=self.torch_device)
                    if isinstance(self.model, nn.Module):
                        self.model.eval()
                    else:
                        # If the loaded object is a dict (checkpoint)
                        if isinstance(self.model, dict) and 'model' in self.model:
                            self.model = self.model['model']
                            self.model.eval()
                            self.model.to(self.torch_device)
            
            # Try to load class names from the same directory
            model_dir = os.path.dirname(self.model_path)
            class_file = os.path.join(model_dir, 'classes.json')
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    self.class_names = json.load(f)
                    logger.info(f"Loaded {len(self.class_names)} class names")
            
            self.is_initialized = True
            logger.info(f"PyTorch model initialized on {self.device}")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing PyTorch model: {str(e)}")
            return False
    
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of preprocessed images.
        
        Args:
            batch: Batch of preprocessed images as numpy array
            
        Returns:
            Batch of predictions (classification probabilities)
        """
        if not self.is_initialized:
            logger.error("Model not initialized. Call initialize() first.")
            return np.zeros((batch.shape[0], self.num_classes))
        
        try:
            # Convert numpy array to PyTorch tensor
            batch_size = batch.shape[0]
            
            # Ensure batch is in the correct format (N, C, H, W)
            if batch.shape[1] != 3 and batch.shape[3] == 3:
                # Convert from (N, H, W, C) to (N, C, H, W)
                batch = np.transpose(batch, (0, 3, 1, 2))
            
            # Create tensor
            with self.lock:
                tensor = self.torch.from_numpy(batch).float().to(self.torch_device)
                
                # Run inference with gradient computation disabled
                with self.torch.no_grad():
                    outputs = self.model(tensor)
                
                # Apply softmax if needed
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if outputs.dim() > 1:
                    # Apply softmax to convert logits to probabilities
                    probs = self.torch.nn.functional.softmax(outputs, dim=1)
                else:
                    probs = outputs
                
                # Convert to numpy
                return probs.cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error during PyTorch inference: {str(e)}")
            return np.zeros((batch_size, self.num_classes))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_initialized:
            return {
                "status": "not_initialized",
                "model_path": self.model_path,
                "device": self.device,
                "quantized": self.quantized
            }
        
        info = {
            "status": "initialized",
            "model_path": self.model_path,
            "device": self.device,
            "quantized": self.quantized,
            "batch_size": self.batch_size,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "class_names": self.class_names,
            "model_type": str(type(self.model))
        }
        
        # Add CUDA information if available
        if self.device == 'cuda':
            try:
                info.update({
                    "cuda_device": self.torch.cuda.get_device_name(0),
                    "cuda_memory_allocated": self.torch.cuda.memory_allocated(),
                    "cuda_memory_cached": self.torch.cuda.memory_reserved()
                })
            except Exception:
                pass
        
        return info
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.is_initialized:
            with self.lock:
                self.model = None
                
                if self.device == 'cuda':
                    try:
                        self.torch.cuda.empty_cache()
                    except Exception:
                        pass
            
            self.is_initialized = False
            logger.info("PyTorch model resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.cleanup()


class TensorflowModelInterface(ModelInterface):
    """TensorFlow model interface."""
    
    def __init__(self,
                 model_path: str,
                 use_gpu: bool = True,
                 batch_size: int = 8,
                 num_classes: int = 8,
                 input_size: Tuple[int, int] = (224, 224)):
        """
        Initialize TensorFlow model interface.
        
        Args:
            model_path: Path to the model file or directory
            use_gpu: Whether to use GPU
            batch_size: Maximum batch size
            num_classes: Number of classes
            input_size: Input image size
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_size = input_size
        
        self.model = None
        self.is_initialized = False
        self.class_names = []
        self.lock = threading.Lock()
    
    def initialize(self) -> bool:
        """Initialize the TensorFlow model."""
        try:
            import tensorflow as tf
            
            self.tf = tf
            
            # Configure GPU memory growth to avoid allocating all memory
            if self.use_gpu:
                gpus = tf.config.experimental.list_physical_devices('GPU')
                if gpus:
                    # Set memory growth for each GPU
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info(f"GPU memory growth enabled for {len(gpus)} GPUs")
                else:
                    logger.warning("No GPUs found, falling back to CPU")
            
            # Load the model
            with self.lock:
                self.model = tf.keras.models.load_model(self.model_path)
            
            # Try to load class names from the same directory
            model_dir = os.path.dirname(self.model_path)
            class_file = os.path.join(model_dir, 'classes.json')
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    self.class_names = json.load(f)
                    logger.info(f"Loaded {len(self.class_names)} class names")
            
            self.is_initialized = True
            logger.info(f"TensorFlow model initialized with GPU: {self.use_gpu}")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing TensorFlow model: {str(e)}")
            return False
    
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of preprocessed images.
        
        Args:
            batch: Batch of preprocessed images as numpy array
            
        Returns:
            Batch of predictions (classification probabilities)
        """
        if not self.is_initialized:
            logger.error("Model not initialized. Call initialize() first.")
            return np.zeros((batch.shape[0], self.num_classes))
        
        try:
            # Ensure batch is in the correct format
            batch_size = batch.shape[0]
            
            with self.lock:
                # Run inference
                predictions = self.model.predict(batch, verbose=0)
                
            return predictions
        
        except Exception as e:
            logger.error(f"Error during TensorFlow inference: {str(e)}")
            return np.zeros((batch_size, self.num_classes))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_initialized:
            return {
                "status": "not_initialized",
                "model_path": self.model_path,
                "use_gpu": self.use_gpu
            }
        
        info = {
            "status": "initialized",
            "model_path": self.model_path,
            "use_gpu": self.use_gpu,
            "batch_size": self.batch_size,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "class_names": self.class_names
        }
        
        # Add model architecture information
        try:
            info["model_summary"] = str(self.model.summary())
        except Exception:
            pass
        
        return info
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.is_initialized:
            with self.lock:
                self.model = None
            
            self.is_initialized = False
            logger.info("TensorFlow model resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.cleanup()


class ONNXModelInterface(ModelInterface):
    """ONNX model interface."""
    
    def __init__(self,
                 model_path: str,
                 use_gpu: bool = False,
                 batch_size: int = 8,
                 num_classes: int = 8,
                 input_size: Tuple[int, int] = (224, 224),
                 input_name: str = None,
                 output_name: str = None):
        """
        Initialize ONNX model interface.
        
        Args:
            model_path: Path to the ONNX model file
            use_gpu: Whether to use GPU
            batch_size: Maximum batch size
            num_classes: Number of classes
            input_size: Input image size
            input_name: Name of the input tensor (if None, will be inferred)
            output_name: Name of the output tensor (if None, will be inferred)
        """
        self.model_path = model_path
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.input_size = input_size
        self.input_name = input_name
        self.output_name = output_name
        
        self.session = None
        self.is_initialized = False
        self.class_names = []
        self.lock = threading.Lock()
    
    def initialize(self) -> bool:
        """Initialize the ONNX model."""
        try:
            import onnxruntime as ort
            
            self.ort = ort
            
            # Setup providers
            providers = ['CPUExecutionProvider']
            if self.use_gpu and 'CUDAExecutionProvider' in ort.get_available_providers():
                providers.insert(0, 'CUDAExecutionProvider')
            
            # Create session
            with self.lock:
                self.session = ort.InferenceSession(self.model_path, providers=providers)
                
                # Get input and output names if not provided
                if self.input_name is None:
                    self.input_name = self.session.get_inputs()[0].name
                
                if self.output_name is None:
                    self.output_name = self.session.get_outputs()[0].name
            
            # Try to load class names from the same directory
            model_dir = os.path.dirname(self.model_path)
            class_file = os.path.join(model_dir, 'classes.json')
            if os.path.exists(class_file):
                with open(class_file, 'r') as f:
                    self.class_names = json.load(f)
                    logger.info(f"Loaded {len(self.class_names)} class names")
            
            self.is_initialized = True
            logger.info(f"ONNX model initialized with providers: {providers}")
            return True
        
        except Exception as e:
            logger.error(f"Error initializing ONNX model: {str(e)}")
            return False
    
    def predict(self, batch: np.ndarray) -> np.ndarray:
        """
        Run inference on a batch of preprocessed images.
        
        Args:
            batch: Batch of preprocessed images as numpy array
            
        Returns:
            Batch of predictions (classification probabilities)
        """
        if not self.is_initialized:
            logger.error("Model not initialized. Call initialize() first.")
            return np.zeros((batch.shape[0], self.num_classes))
        
        try:
            # Ensure batch is in the correct format (N, C, H, W) for ONNX
            batch_size = batch.shape[0]
            
            # Convert from (N, H, W, C) to (N, C, H, W) if needed
            if batch.shape[1] != 3 and batch.shape[3] == 3:
                batch = np.transpose(batch, (0, 3, 1, 2))
            
            # Run inference
            with self.lock:
                outputs = self.session.run(
                    [self.output_name],
                    {self.input_name: batch.astype(np.float32)}
                )
                
            # Process the output
            predictions = outputs[0]
            
            # Apply softmax if needed
            if np.max(predictions) > 100 or np.min(predictions) < -100:
                # These are likely logits, apply softmax
                # Shift for numerical stability
                predictions = predictions - np.max(predictions, axis=1, keepdims=True)
                exp_preds = np.exp(predictions)
                predictions = exp_preds / np.sum(exp_preds, axis=1, keepdims=True)
            
            return predictions
        
        except Exception as e:
            logger.error(f"Error during ONNX inference: {str(e)}")
            return np.zeros((batch_size, self.num_classes))
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information."""
        if not self.is_initialized:
            return {
                "status": "not_initialized",
                "model_path": self.model_path,
                "use_gpu": self.use_gpu
            }
        
        info = {
            "status": "initialized",
            "model_path": self.model_path,
            "use_gpu": self.use_gpu,
            "batch_size": self.batch_size,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "class_names": self.class_names,
            "input_name": self.input_name,
            "output_name": self.output_name,
            "providers": self.session.get_providers()
        }
        
        return info
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.is_initialized:
            with self.lock:
                self.session = None
            
            self.is_initialized = False
            logger.info("ONNX model resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure resources are cleaned up."""
        self.cleanup()


class ModelFactory:
    """Factory class for creating model interfaces."""
    
    @staticmethod
    def create_model(model_type: str, **kwargs) -> ModelInterface:
        """
        Create a model interface based on the given type.
        
        Args:
            model_type: Type of model ('pytorch', 'tensorflow', 'onnx')
            **kwargs: Additional parameters for the specific model
            
        Returns:
            A ModelInterface implementation
            
        Raises:
            ValueError: If the model type is not supported
        """
        if model_type.lower() == 'pytorch':
            return PyTorchModelInterface(**kwargs)
        elif model_type.lower() in ['tensorflow', 'tf']:
            return TensorflowModelInterface(**kwargs)
        elif model_type.lower() == 'onnx':
            return ONNXModelInterface(**kwargs)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")


class TrashClassifier:
    """Main trash classifier using the model interface."""
    
    # Mapping of class indices to trash categories
    DEFAULT_CLASS_MAPPING = {
        0: 'recyclable_plastic',
        1: 'recyclable_glass',
        2: 'recyclable_metal',
        3: 'recyclable_paper',
        4: 'compostable',
        5: 'landfill',
        6: 'ewaste',
        7: 'unknown'
    }
    
    def __init__(self, 
                 model: ModelInterface,
                 batch_size: int = 8,
                 confidence_threshold: float = 0.7,
                 class_mapping: Dict[int, str] = None):
        """
        Initialize trash classifier.
        
        Args:
            model: Model interface
            batch_size: Batch size for inference
            confidence_threshold: Confidence threshold for classification
            class_mapping: Mapping of class indices to trash categories
        """
        self.model = model
        self.batch_size = batch_size
        self.confidence_threshold = confidence_threshold
        self.class_mapping = class_mapping or self.DEFAULT_CLASS_MAPPING
        
        self.initialized = False
    
    def initialize(self) -> bool:
        """Initialize the trash classifier."""
        if self.model.initialize():
            self.initialized = True
            
            # Check if model has class names
            model_info = self.model.get_model_info()
            if 'class_names' in model_info and model_info['class_names']:
                # Create class mapping from model's class names
                self.class_mapping = {
                    i: name.lower().replace(' ', '_')
                    for i, name in enumerate(model_info['class_names'])
                }
                logger.info(f"Using class mapping from model: {self.class_mapping}")
            
            return True
        
        return False
    
    def predict_batch(self, batch: np.ndarray) -> List[Dict[str, Any]]:
        """
        Predict trash categories for a batch of preprocessed images.
        
        Args:
            batch: Batch of preprocessed images
            
        Returns:
            List of prediction results
        """
        if not self.initialized:
            logger.error("Trash classifier not initialized. Call initialize() first.")
            return []
        
        # Run inference
        predictions = self.model.predict(batch)
        
        # Process predictions
        results = []
        for i, pred in enumerate(predictions):
            # Get the class index and confidence
            class_idx = np.argmax(pred)
            confidence = float(pred[class_idx])
            
            # Get the category
            if confidence >= self.confidence_threshold:
                category = self.class_mapping.get(class_idx, 'unknown')
            else:
                category = 'unknown'
                
            results.append({
                'category': category,
                'confidence': confidence,
                'class_idx': int(class_idx),
                'probabilities': {
                    self.class_mapping.get(j, f'class_{j}'): float(p)
                    for j, p in enumerate(pred)
                }
            })
        
        return results
    
    def predict_single(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Predict trash category for a single preprocessed image.
        
        Args:
            image: Preprocessed image
            
        Returns:
            Prediction result
        """
        # Create a batch with a single image
        batch = np.expand_dims(image, axis=0)
        
        # Run prediction
        results = self.predict_batch(batch)
        
        return results[0] if results else {}
    
    def batch_average(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Average predictions across a batch.
        
        Args:
            batch_results: List of prediction results
            
        Returns:
            Averaged prediction result
        """
        if not batch_results:
            return {}
        
        # Collect all probabilities
        all_probs = {}
        for result in batch_results:
            for class_name, prob in result['probabilities'].items():
                if class_name not in all_probs:
                    all_probs[class_name] = []
                all_probs[class_name].append(prob)
        
        # Average probabilities
        avg_probs = {
            class_name: np.mean(probs)
            for class_name, probs in all_probs.items()
        }
        
        # Get the most likely category
        max_class = max(avg_probs.items(), key=lambda x: x[1])
        category = max_class[0]
        confidence = max_class[1]
        
        # Apply confidence threshold
        if confidence < self.confidence_threshold:
            category = 'unknown'
        
        return {
            'category': category,
            'confidence': float(confidence),
            'averaged': True,
            'probabilities': {
                class_name: float(prob)
                for class_name, prob in avg_probs.items()
            }
        }
    
    def cleanup(self) -> None:
        """Clean up resources."""
        if self.initialized:
            self.model.cleanup()
            self.initialized = False
            logger.info("Trash classifier resources cleaned up")
    
    def __enter__(self):
        """Context manager enter method."""
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit method."""
        self.cleanup()


# Example usage
if __name__ == "__main__":
    import cv2
    from time import time
    
    # Example: Using PyTorch model for trash classification
    model_path = "models/trash_classifier_resnet18.pt"
    
    try:
        # Create model interface
        model = ModelFactory.create_model(
            'pytorch',
            model_path=model_path,
            device='cuda',
            batch_size=8,
            num_classes=8
        )
        
        # Create trash classifier
        classifier = TrashClassifier(
            model=model,
            batch_size=8,
            confidence_threshold=0.7
        )
        
        if classifier.initialize():
            print("Trash classifier initialized successfully")
            
            # Load and preprocess test image
            image_path = "test_image.jpg"
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not load image: {image_path}")
            
            # Preprocess image (simple resize for example)
            preprocessed = cv2.resize(image, (224, 224))
            preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2RGB)
            preprocessed = preprocessed.astype(np.float32) / 255.0
            
            # Predict
            start_time = time()
            result = classifier.predict_single(preprocessed)
            inference_time = time() - start_time
            
            print(f"Prediction: {result['category']}")
            print(f"Confidence: {result['confidence']:.2f}")
            print(f"Inference time: {inference_time*1000:.2f} ms")
            
            # Create a batch for batch inference
            batch = np.array([preprocessed] * 8)
            
            # Predict batch
            start_time = time()
            batch_results = classifier.predict_batch(batch)
            batch_inference_time = time() - start_time
            
            print(f"Batch inference time: {batch_inference_time*1000:.2f} ms")
            print(f"Average inference time per image: {batch_inference_time*1000/8:.2f} ms")
            
            # Average results
            avg_result = classifier.batch_average(batch_results)
            print(f"Averaged prediction: {avg_result['category']}")
            print(f"Averaged confidence: {avg_result['confidence']:.2f}")
        
        else:
            print("Failed to initialize trash classifier")
    
    except Exception as e:
        print(f"Error: {str(e)}")
    
    finally:
        # Clean up
        if 'classifier' in locals():
            classifier.cleanup()
            print("Trash classifier resources cleaned up") 