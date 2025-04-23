#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ResNet model implementation for trash classification.
Provides fine-tuning of pretrained ResNet models for trash classification.
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


class ResNetModel(nn.Module):
    """ResNet model for trash classification."""
    
    def __init__(self, 
                 num_classes: int = 8,
                 model_version: str = '50',
                 pretrained: bool = True,
                 dropout_rate: float = 0.3,
                 feature_extract: bool = True):
        """
        Initialize ResNet model.
        
        Args:
            num_classes: Number of output classes
            model_version: ResNet version ('18', '34', '50', '101', '152')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for final classifier
            feature_extract: Whether to use feature extraction (frozen backbone) or fine-tuning
        """
        super(ResNetModel, self).__init__()
        
        self.num_classes = num_classes
        self.model_version = model_version
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.feature_extract = feature_extract
        
        # Load base model
        self._initialize_base_model()
        
        # Modify final layer for classification
        self._modify_classifier()
        
        logger.info(f"Initialized ResNet-{model_version} model with {num_classes} output classes")
        logger.info(f"Pretrained: {pretrained}, Feature extraction: {feature_extract}")
    
    def _initialize_base_model(self):
        """Initialize the base ResNet model."""
        # Map model version to corresponding torchvision function
        model_map = {
            '18': models.resnet18,
            '34': models.resnet34,
            '50': models.resnet50,
            '101': models.resnet101,
            '152': models.resnet152,
        }
        
        # Check if the model version is valid
        if self.model_version not in model_map:
            raise ValueError(f"Invalid model version: {self.model_version}")
        
        # Load model with appropriate weights
        weights = None
        if self.pretrained:
            weights_map = {
                '18': models.ResNet18_Weights.DEFAULT,
                '34': models.ResNet34_Weights.DEFAULT,
                '50': models.ResNet50_Weights.DEFAULT,
                '101': models.ResNet101_Weights.DEFAULT,
                '152': models.ResNet152_Weights.DEFAULT,
            }
            weights = weights_map[self.model_version]
                
        self.model = model_map[self.model_version](weights=weights)
        
        # Freeze parameters if using feature extraction
        if self.feature_extract:
            for param in self.model.parameters():
                param.requires_grad = False
    
    def _modify_classifier(self):
        """Modify the final classifier layer for the number of classes."""
        # Get the number of input features
        num_ftrs = self.model.fc.in_features
        
        # Replace the final classification layer with optional dropout
        if self.dropout_rate > 0:
            self.model.fc = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.Linear(num_ftrs, self.num_classes)
            )
        else:
            self.model.fc = nn.Linear(num_ftrs, self.num_classes)
    
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
            'dropout_rate': self.dropout_rate,
            'model_type': 'ResNet',
            'input_size': (224, 224)  # ResNet uses 224x224 images by default
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
    def load(path: str, device: torch.device = torch.device('cpu')) -> 'ResNetModel':
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
        model = ResNetModel(
            num_classes=model_info['num_classes'],
            model_version=model_info['model_version'],
            pretrained=False,  # No need to load pretrained weights
            dropout_rate=model_info.get('dropout_rate', 0.3),
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
    model = ResNetModel(
        num_classes=8,
        model_version='50',
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
    model.save('resnet_model.pth')
    
    # Export to ONNX
    model.to_onnx('resnet_model.onnx') 