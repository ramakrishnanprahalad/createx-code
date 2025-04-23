#!/usr/bin/env python3
import os
import shutil
import random
import sys

# Set random seed for reproducibility
random.seed(42)

# Get current script directory
script_dir = os.path.dirname(os.path.abspath(__file__))
print(f"Script directory: {script_dir}")

# Ensure script directory path is correct
if not os.path.exists(script_dir):
    print(f"Error: Script directory {script_dir} does not exist")
    sys.exit(1)

# Print current working directory
print(f"Current working directory: {os.getcwd()}")

# Paths
source_dir = os.path.join(script_dir, 'dataset-resized')
parent_dir = os.path.dirname(os.path.dirname(script_dir))  # Go up two levels
output_dir = os.path.join(parent_dir, 'data', 'trash_classification')
train_dir = os.path.join(output_dir, 'train')
val_dir = os.path.join(output_dir, 'val')
test_dir = os.path.join(output_dir, 'test')

# Check if source directory exists
if not os.path.exists(source_dir):
    print(f"Error: Source directory {source_dir} does not exist")
    sys.exit(1)

# Split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Categories
categories = ['glass', 'paper', 'cardboard', 'plastic', 'metal', 'trash']

# Print paths for debugging
print(f"Source directory: {source_dir}")
print(f"Parent directory: {parent_dir}")
print(f"Output directory: {output_dir}")
print(f"Train directory: {train_dir}")
print(f"Val directory: {val_dir}")
print(f"Test directory: {test_dir}")

# Check if all categories exist in the source directory
for category in categories:
    category_path = os.path.join(source_dir, category)
    if not os.path.exists(category_path):
        print(f"Error: Category directory {category_path} does not exist")
        sys.exit(1)

# Create directories if they don't exist
for category in categories:
    os.makedirs(os.path.join(train_dir, category), exist_ok=True)
    os.makedirs(os.path.join(val_dir, category), exist_ok=True)
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

# Function to split and copy files
def split_files(category):
    # List all image files in the source directory
    source_path = os.path.join(source_dir, category)
    files = [f for f in os.listdir(source_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    print(f"Found {len(files)} files in {source_path}")
    
    # Shuffle files to ensure randomness
    random.shuffle(files)
    
    # Calculate split indices
    n_files = len(files)
    n_train = int(train_ratio * n_files)
    n_val = int(val_ratio * n_files)
    
    train_files = files[:n_train]
    val_files = files[n_train:n_train+n_val]
    test_files = files[n_train+n_val:]
    
    # Copy files to respective directories
    count = 0
    for file in train_files:
        try:
            source_file = os.path.join(source_path, file)
            dest_file = os.path.join(train_dir, category, file)
            shutil.copy(source_file, dest_file)
            count += 1
            if count % 50 == 0:
                print(f"Copied {count} files to train directory")
        except Exception as e:
            print(f"Error copying {source_file} to {dest_file}: {e}")
    
    count = 0
    for file in val_files:
        try:
            source_file = os.path.join(source_path, file)
            dest_file = os.path.join(val_dir, category, file)
            shutil.copy(source_file, dest_file)
            count += 1
            if count % 50 == 0:
                print(f"Copied {count} files to val directory")
        except Exception as e:
            print(f"Error copying {source_file} to {dest_file}: {e}")
    
    count = 0
    for file in test_files:
        try:
            source_file = os.path.join(source_path, file)
            dest_file = os.path.join(test_dir, category, file)
            shutil.copy(source_file, dest_file)
            count += 1
            if count % 50 == 0:
                print(f"Copied {count} files to test directory")
        except Exception as e:
            print(f"Error copying {source_file} to {dest_file}: {e}")
    
    print(f"{category}: Total={n_files}, Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")

# Process each category
for category in categories:
    split_files(category)

# Verify files were copied
for category in categories:
    train_path = os.path.join(train_dir, category)
    val_path = os.path.join(val_dir, category)
    test_path = os.path.join(test_dir, category)
    train_count = len(os.listdir(train_path))
    val_count = len(os.listdir(val_path))
    test_count = len(os.listdir(test_path))
    print(f"Verification - {category}: Train={train_count}, Val={val_count}, Test={test_count}")

print("Dataset splitting completed.") 