#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MobileNet model implementation for trash classification.
Provides fine-tuning of pretrained MobileNet models for trash classification.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MobileNetModel(nn.Module):
    """MobileNet model for trash classification."""
    
    def __init__(self, 
                 num_classes: int = 8,
                 model_version: str = 'v2',
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 feature_extract: bool = True):
        """
        Initialize MobileNet model.
        
        Args:
            num_classes: Number of output classes
            model_version: MobileNet version ('v2' or 'v3_small' or 'v3_large')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for final classifier
            feature_extract: Whether to use feature extraction (frozen backbone) or fine-tuning
        """
        super(MobileNetModel, self).__init__()
        
        self.num_classes = num_classes
        self.model_version = model_version
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.feature_extract = feature_extract
        
        # Load base model
        self._initialize_base_model()
        
        # Modify final layer for classification
        self._modify_classifier()
        
        logger.info(f"Initialized MobileNet-{model_version} model with {num_classes} output classes")
        logger.info(f"Pretrained: {pretrained}, Feature extraction: {feature_extract}")
    
    def _initialize_base_model(self):
        """Initialize the base MobileNet model."""
        # Map model version to corresponding torchvision function
        model_map = {
            'v2': models.mobilenet_v2,
            'v3_small': models.mobilenet_v3_small,
            'v3_large': models.mobilenet_v3_large
        }
        
        # Check if the model version is valid
        if self.model_version not in model_map:
            raise ValueError(f"Invalid model version: {self.model_version}")
        
        # Load model with appropriate weights
        weights = None
        if self.pretrained:
            weights_map = {
                'v2': models.MobileNet_V2_Weights.DEFAULT,
                'v3_small': models.MobileNet_V3_Small_Weights.DEFAULT,
                'v3_large': models.MobileNet_V3_Large_Weights.DEFAULT
            }
            weights = weights_map[self.model_version]
                
        self.model = model_map[self.model_version](weights=weights)
        
        # Freeze parameters if using feature extraction
        if self.feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def _modify_classifier(self):
        """Modify the final classifier layer for the number of classes."""
        # Get the final classifier
        if self.model_version == 'v2':
            num_ftrs = self.model.classifier[1].in_features
            self.model.classifier = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(num_ftrs, self.num_classes)
            )
        else:  # v3_small or v3_large
            num_ftrs = self.model.classifier[3].in_features
            self.model.classifier[3] = nn.Linear(num_ftrs, self.num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, num_classes)
        """
        return self.model(x)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_version': self.model_version,
            'num_classes': self.num_classes,
            'pretrained': self.pretrained,
            'feature_extract': self.feature_extract,
            'model_type': 'MobileNet',
            'input_size': (224, 224)  # MobileNet uses 224x224 images by default
        }
    
    def save(self, path: str):
        """
        Save model weights.
        
        Args:
            path: Path to save the model weights
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model
        torch.save({
            'model_state_dict': self.state_dict(),
            'model_info': self.get_model_info()
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    @staticmethod
    def load(path: str, device: torch.device = torch.device('cpu')) -> 'MobileNetModel':
        """
        Load model from file.
        
        Args:
            path: Path to the model file
            device: Device to load the model to
            
        Returns:
            Loaded model
        """
        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Get model info
        model_info = checkpoint['model_info']
        
        # Create model
        model = MobileNetModel(
            num_classes=model_info['num_classes'],
            model_version=model_info['model_version'],
            pretrained=False,  # No need to load pretrained weights
            feature_extract=model_info['feature_extract']
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Move model to device
        model = model.to(device)
        
        logger.info(f"Model loaded from {path}")
        
        return model
    
    def to_onnx(self, path: str, input_shape: Optional[Tuple[int, int, int, int]] = None):
        """
        Export model to ONNX format.
        
        Args:
            path: Path to save the ONNX model
            input_shape: Input shape (batch_size, channels, height, width)
        """
        # Get input size if not provided
        if input_shape is None:
            model_info = self.get_model_info()
            height, width = model_info['input_size']
            input_shape = (1, 3, height, width)
        
        # Create input tensor
        dummy_input = torch.randn(input_shape)
        
        # Export model
        torch.onnx.export(
            self,
            dummy_input,
            path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        logger.info(f"Model exported to ONNX format at {path}")


# Example usage
if __name__ == "__main__":
    # Create model
    model = MobileNetModel(
        num_classes=8,
        model_version='v2',
        pretrained=True,
        feature_extract=True
    )
    
    # Create dummy input
    model_info = model.get_model_info()
    height, width = model_info['input_size']
    dummy_input = torch.randn(1, 3, height, width)
    
    # Forward pass
    output = model(dummy_input)
    
    # Print output shape
    print(f"Output shape: {output.shape}")
    
    # Get model info
    print(f"Model info: {model_info}")
    
    # Save model
    model.save('mobilenet_model.pth')
    
    # Export to ONNX
    model.to_onnx('mobilenet_model.onnx') 