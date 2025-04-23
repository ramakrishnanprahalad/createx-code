#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VGG model implementation for trash classification.
Provides fine-tuning of pretrained VGG models for trash classification.
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


class VGGModel(nn.Module):
    """VGG model for trash classification."""
    
    def __init__(self, 
                 num_classes: int = 8,
                 model_version: str = '16',
                 pretrained: bool = True,
                 dropout_rate: float = 0.5,
                 feature_extract: bool = True):
        """
        Initialize VGG model.
        
        Args:
            num_classes: Number of output classes
            model_version: VGG version ('11', '13', '16', '19')
            pretrained: Whether to use pretrained weights
            dropout_rate: Dropout rate for final classifier
            feature_extract: Whether to use feature extraction (frozen backbone) or fine-tuning
        """
        super(VGGModel, self).__init__()
        
        self.num_classes = num_classes
        self.model_version = model_version
        self.pretrained = pretrained
        self.dropout_rate = dropout_rate
        self.feature_extract = feature_extract
        
        # Load base model
        self._initialize_base_model()
        
        # Modify final layer for classification
        self._modify_classifier()
        
        logger.info(f"Initialized VGG-{model_version} model with {num_classes} output classes")
        logger.info(f"Pretrained: {pretrained}, Feature extraction: {feature_extract}")
    
    def _initialize_base_model(self):
        """Initialize the base VGG model."""
        # Map model version to corresponding torchvision function
        model_map = {
            '11': models.vgg11,
            '13': models.vgg13,
            '16': models.vgg16,
            '19': models.vgg19,
            '11_bn': models.vgg11_bn,
            '13_bn': models.vgg13_bn,
            '16_bn': models.vgg16_bn,
            '19_bn': models.vgg19_bn,
        }
        
        # Check if the model version is valid
        if self.model_version not in model_map:
            raise ValueError(f"Invalid model version: {self.model_version}")
        
        # Load model with appropriate weights
        weights = None
        if self.pretrained:
            weights_map = {
                '11': models.VGG11_Weights.DEFAULT,
                '13': models.VGG13_Weights.DEFAULT,
                '16': models.VGG16_Weights.DEFAULT,
                '19': models.VGG19_Weights.DEFAULT,
                '11_bn': models.VGG11_BN_Weights.DEFAULT,
                '13_bn': models.VGG13_BN_Weights.DEFAULT,
                '16_bn': models.VGG16_BN_Weights.DEFAULT,
                '19_bn': models.VGG19_BN_Weights.DEFAULT,
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
        num_ftrs = self.model.classifier[6].in_features
        
        # Replace the final classification layer
        self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)
        
        # Update dropout rate if different from default
        if self.dropout_rate != 0.5:  # VGG default dropout is 0.5
            self.model.classifier[2] = nn.Dropout(p=self.dropout_rate)
            self.model.classifier[5] = nn.Dropout(p=self.dropout_rate)
    
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
            'model_type': 'VGG',
            'input_size': (224, 224)  # VGG uses 224x224 images by default
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
    def load(path: str, device: torch.device = torch.device('cpu')) -> 'VGGModel':
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
        model = VGGModel(
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
    model = VGGModel(
        num_classes=8,
        model_version='16',
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
    model.save('vgg_model.pth')
    
    # Export to ONNX
    model.to_onnx('vgg_model.onnx') 