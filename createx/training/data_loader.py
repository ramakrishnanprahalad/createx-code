#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Data loading and augmentation module for trash classification.
Handles dataset loading, augmentation, and preprocessing.
"""

import os
import json
import random
import logging
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_train_val_transforms(
    img_size: Tuple[int, int] = (224, 224),
    mean: List[float] = [0.485, 0.456, 0.406],
    std: List[float] = [0.229, 0.224, 0.225]
) -> Tuple[A.Compose, A.Compose]:
    """
    Create training and validation transforms.
    
    Args:
        img_size: Target image size
        mean: Mean values for normalization (ImageNet default)
        std: Standard deviation values for normalization (ImageNet default)
        
    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    train_transforms = A.Compose([
        A.RandomResizedCrop(height=img_size[0], width=img_size[1], scale=(0.8, 1.0)),
        A.Rotate(limit=30, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomBrightnessContrast(p=0.7),
        A.HueSaturationValue(p=0.5),
        A.GaussNoise(p=0.3),
        A.OneOf([
            A.MotionBlur(p=0.7),
            A.MedianBlur(blur_limit=3, p=0.5),
            A.Blur(blur_limit=3, p=0.5),
        ], p=0.4),
        A.OneOf([
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=0.5),
            A.ElasticTransform(p=0.5),
        ], p=0.2),
        A.CoarseDropout(max_holes=4, max_height=img_size[0]//10, max_width=img_size[1]//10, p=0.4),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    val_transforms = A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return train_transforms, val_transforms


class TrashDataset(Dataset):
    """Dataset for trash classification."""
    
    def __init__(self, 
                 data_dir: str, 
                 transforms: Optional[A.Compose] = None,
                 split: str = 'train'):
        """
        Initialize trash dataset.
        
        Args:
            data_dir: Path to dataset directory
            transforms: Albumentations transforms pipeline
            split: Dataset split ('train', 'val', 'test')
        """
        self.data_dir = Path(data_dir)
        self.transforms = transforms
        self.split = split
        
        # Load dataset
        self.images = []
        self.labels = []
        self.class_names = []
        self.class_to_idx = {}
        
        # Check if dataset exists
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        
        # Load class names
        self._load_classes()
        
        # Load data
        self._load_data()
        
        logger.info(f"Loaded {len(self.images)} images for {split} dataset")
        logger.info(f"Found {len(self.class_names)} classes: {self.class_names}")
    
    def _load_classes(self):
        """Load class names and mapping."""
        # Find class directories
        class_dirs = [d for d in self.data_dir.iterdir() if d.is_dir()]
        
        # Sort to ensure consistent ordering
        class_dirs.sort()
        
        self.class_names = [d.name for d in class_dirs]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}
        
        # Save class mapping to JSON
        class_json = self.data_dir / 'classes.json'
        with open(class_json, 'w') as f:
            json.dump(self.class_names, f)
    
    def _load_data(self):
        """Load dataset files and labels."""
        # Process each class directory
        for class_name in self.class_names:
            class_dir = self.data_dir / class_name
            
            # Get all image files
            valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
            for img_path in class_dir.glob('*'):
                if img_path.suffix.lower() in valid_extensions:
                    # Add image and label
                    self.images.append(str(img_path))
                    self.labels.append(self.class_to_idx[class_name])
    
    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Dict containing image and label tensors
        """
        # Get image path and label
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Read image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed['image']
        
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        }


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    img_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4,
    val_split: float = 0.2,
    test_split: float = 0.1
) -> Dict[str, DataLoader]:
    """
    Create dataloaders for training, validation and testing.
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size
        img_size: Image size for resizing and cropping
        num_workers: Number of worker processes for data loading
        val_split: Validation split ratio (0.0-1.0)
        test_split: Test split ratio (0.0-1.0)
        
    Returns:
        Dictionary of dataloaders for each split
    """
    # Create transforms
    train_transforms, val_transforms = create_train_val_transforms(img_size=img_size)
    
    # Create dataset
    full_dataset = TrashDataset(data_dir, transforms=None)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    val_size = int(val_split * total_size)
    test_size = int(test_split * total_size)
    train_size = total_size - val_size - test_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Apply transforms
    train_dataset.dataset.transforms = train_transforms
    val_dataset.dataset.transforms = val_transforms
    test_dataset.dataset.transforms = val_transforms
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders


class SyntheticTrashDataset(Dataset):
    """
    Synthetic dataset generator for trash classification.
    Useful for development and testing when real data is not available.
    """
    
    def __init__(self, 
                 num_classes: int = 8,
                 num_samples: int = 1000,
                 img_size: Tuple[int, int] = (224, 224),
                 transforms: Optional[A.Compose] = None):
        """
        Initialize synthetic dataset.
        
        Args:
            num_classes: Number of classes
            num_samples: Number of samples to generate
            img_size: Image size
            transforms: Albumentations transforms pipeline
        """
        self.num_classes = num_classes
        self.num_samples = num_samples
        self.img_size = img_size
        self.transforms = transforms
        
        # Generate sample data
        self.labels = torch.randint(0, num_classes, (num_samples,))
        
        # Create class names
        self.class_names = [
            'recyclable_plastic',
            'recyclable_glass',
            'recyclable_metal',
            'recyclable_paper',
            'compostable',
            'landfill',
            'ewaste',
            'unknown'
        ][:num_classes]
        
        # Colors for each class to make synthetic data more realistic
        self.class_colors = [
            (0, 0, 255),    # Blue for plastic
            (0, 255, 255),  # Cyan for glass
            (192, 192, 192),  # Silver for metal
            (255, 255, 224),  # Light yellow for paper
            (0, 128, 0),    # Green for compostable
            (128, 128, 128),  # Gray for landfill
            (128, 0, 128),  # Purple for ewaste
            (64, 64, 64),   # Dark gray for unknown
        ][:num_classes]
    
    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get dataset item by index.
        
        Args:
            idx: Item index
            
        Returns:
            Dict containing image and label tensors
        """
        # Get label
        label = self.labels[idx].item()
        
        # Create synthetic image
        img = self._generate_synthetic_image(label)
        
        # Apply transforms
        if self.transforms:
            transformed = self.transforms(image=img)
            img = transformed['image']
        
        return {
            'image': img,
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def _generate_synthetic_image(self, label: int) -> np.ndarray:
        """
        Generate a synthetic image for the given label.
        
        Args:
            label: Class label
            
        Returns:
            Synthetic image
        """
        # Create base image with class color
        base_color = self.class_colors[label]
        img = np.ones((self.img_size[0], self.img_size[1], 3), dtype=np.uint8) * base_color
        
        # Add some randomness
        noise = np.random.randint(0, 100, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)
        
        # Add shapes based on class
        num_shapes = np.random.randint(1, 10)
        for _ in range(num_shapes):
            shape_type = np.random.randint(0, 3)  # 0: circle, 1: rectangle, 2: line
            
            # Random position
            x = np.random.randint(0, self.img_size[1])
            y = np.random.randint(0, self.img_size[0])
            
            # Random size
            size = np.random.randint(10, 100)
            
            # Random color
            color = tuple(np.random.randint(0, 255, 3).tolist())
            
            # Draw shape
            if shape_type == 0:  # Circle
                cv2.circle(img, (x, y), size // 2, color, thickness=np.random.randint(-1, 10))
            elif shape_type == 1:  # Rectangle
                x2 = min(x + size, self.img_size[1] - 1)
                y2 = min(y + size, self.img_size[0] - 1)
                cv2.rectangle(img, (x, y), (x2, y2), color, thickness=np.random.randint(-1, 10))
            else:  # Line
                x2 = min(x + size, self.img_size[1] - 1)
                y2 = min(y + size, self.img_size[0] - 1)
                cv2.line(img, (x, y), (x2, y2), color, thickness=np.random.randint(1, 10))
        
        return img


def create_synthetic_dataloaders(
    num_classes: int = 8,
    num_train_samples: int = 1000,
    num_val_samples: int = 200,
    num_test_samples: int = 100,
    batch_size: int = 32,
    img_size: Tuple[int, int] = (224, 224),
    num_workers: int = 4
) -> Dict[str, DataLoader]:
    """
    Create dataloaders with synthetic data for training, validation and testing.
    
    Args:
        num_classes: Number of classes
        num_train_samples: Number of training samples
        num_val_samples: Number of validation samples
        num_test_samples: Number of test samples
        batch_size: Batch size
        img_size: Image size
        num_workers: Number of worker processes for data loading
        
    Returns:
        Dictionary of dataloaders for each split
    """
    # Create transforms
    train_transforms, val_transforms = create_train_val_transforms(img_size=img_size)
    
    # Create datasets
    train_dataset = SyntheticTrashDataset(
        num_classes=num_classes,
        num_samples=num_train_samples,
        img_size=img_size,
        transforms=train_transforms
    )
    
    val_dataset = SyntheticTrashDataset(
        num_classes=num_classes,
        num_samples=num_val_samples,
        img_size=img_size,
        transforms=val_transforms
    )
    
    test_dataset = SyntheticTrashDataset(
        num_classes=num_classes,
        num_samples=num_test_samples,
        img_size=img_size,
        transforms=val_transforms
    )
    
    # Create dataloaders
    dataloaders = {
        'train': DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        ),
        'val': DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        ),
        'test': DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
    }
    
    return dataloaders


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Create synthetic dataloaders
    dataloaders = create_synthetic_dataloaders(
        num_classes=8,
        num_train_samples=100,
        num_val_samples=20,
        num_test_samples=10,
        batch_size=4
    )
    
    # Get a batch from the training dataloader
    batch = next(iter(dataloaders['train']))
    
    # Display images and labels
    images = batch['image']
    labels = batch['label']
    
    plt.figure(figsize=(12, 8))
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        img = images[i].permute(1, 2, 0).numpy()
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(f"Class: {labels[i].item()}")
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('synthetic_data_sample.png')
    print("Saved synthetic data sample to synthetic_data_sample.png") 