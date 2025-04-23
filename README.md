# Createx - Smart Waste Sorting System

A computer vision-based system for waste classification using deep learning models.

## Overview

Createx is a waste classification system that uses deep learning models to identify different types of trash (cardboard, glass, metal, paper, plastic, and other waste). The system can be integrated with automated waste sorting machines to improve recycling efficiency.

## Project Structure

```
createx/
├── core/              # Core system functionality
│   ├── camera.py      # Camera/image capture functions
│   ├── inference.py   # Model inference code
│   ├── led_control.py # Control LED indicators
│   ├── main.py        # Main application
│   └── preprocessing.py # Image preprocessing
├── data/              # Data directory
│   └── trash_classification/ # Dataset for training models
│       ├── train/     # Training data
│       ├── val/       # Validation data
│       └── test/      # Test data
├── evaluation/        # Model evaluation code
├── training/          # Training modules
│   ├── data_loader.py # Data loading utilities
│   └── models/        # Model implementations
│       ├── densenet_model.py
│       ├── efficientnet_model.py
│       ├── mobilenet_model.py
│       ├── resnet_model.py
│       ├── vgg_model.py
│       └── vit_model.py
└── utils/             # Utility functions
```

## Models

The system includes several deep learning architectures:

- ResNet
- VGG
- MobileNet
- Vision Transformer (ViT)
- DenseNet
- EfficientNet

## Dataset

The project uses the TrashNet dataset, which contains images of waste in six categories:
- Cardboard (403 images)
- Glass (501 images)
- Metal (410 images)
- Paper (594 images)
- Plastic (482 images)
- Trash (137 images)

The dataset has been split into train/validation/test sets with a 70/15/15 ratio.