#!/usr/bin/env python3
import os
import shutil
import random

# Set the seed for reproducibility
random.seed(42)

# Define paths (using absolute paths to avoid any issues)
base_dir = '/Users/prahaladramakrishnan/Desktop/createx-code/createx/data/trash_classification'
source_dir = os.path.join(base_dir, 'trashnet/data/dataset-resized')
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')

# Print paths for debugging
print(f"Source directory: {source_dir}")
print(f"Train directory: {train_dir}")
print(f"Val directory: {val_dir}")
print(f"Test directory: {test_dir}")

# Define categories
categories = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

# Train/val/test split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create destination directories
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Process each category
for category in categories:
    # Get list of all files in this category
    category_dir = os.path.join(source_dir, category)
    files = [f for f in os.listdir(category_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"{category}: Found {len(files)} files")
    
    # Shuffle the files
    random.shuffle(files)
    
    # Split into train, val, test
    num_files = len(files)
    num_train = int(train_ratio * num_files)
    num_val = int(val_ratio * num_files)
    
    train_files = files[:num_train]
    val_files = files[num_train:num_train+num_val]
    test_files = files[num_train+num_val:]
    
    print(f"{category}: Splitting into {len(train_files)} train, {len(val_files)} val, {len(test_files)} test")
    
    # Copy files to their respective directories
    for file in train_files:
        src = os.path.join(category_dir, file)
        dst = os.path.join(train_dir, category, file)
        shutil.copy2(src, dst)
    
    for file in val_files:
        src = os.path.join(category_dir, file)
        dst = os.path.join(val_dir, category, file)
        shutil.copy2(src, dst)
    
    for file in test_files:
        src = os.path.join(category_dir, file)
        dst = os.path.join(test_dir, category, file)
        shutil.copy2(src, dst)

# Verify the result
for category in categories:
    train_count = len(os.listdir(os.path.join(train_dir, category)))
    val_count = len(os.listdir(os.path.join(val_dir, category)))
    test_count = len(os.listdir(os.path.join(test_dir, category)))
    
    print(f"Verification - {category}: {train_count} train, {val_count} val, {test_count} test") 