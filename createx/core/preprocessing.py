#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image preprocessing module for trash classification system.
Contains functions for image preprocessing and object detection.
"""

import cv2
import time
import numpy as np
import logging
from typing import List, Tuple, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ImagePreprocessor:
    """Class for image preprocessing operations."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 normalize: bool = True,
                 mean: List[float] = [0.485, 0.456, 0.406],  # ImageNet mean
                 std: List[float] = [0.229, 0.224, 0.225]):  # ImageNet std
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target image size for model input as (width, height)
            normalize: Whether to normalize pixel values
            mean: Mean values for each channel for normalization
            std: Standard deviation values for each channel for normalization
        """
        self.target_size = target_size
        self.normalize = normalize
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
    
    def resize(self, image: np.ndarray) -> np.ndarray:
        """
        Resize the image to the target size.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Resized image
        """
        return cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values.
        
        Args:
            image: Input image as numpy array (uint8)
            
        Returns:
            Normalized image (float32)
        """
        image = image.astype(np.float32) / 255.0
        image = (image - self.mean) / self.std
        return image
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model input.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image ready for model input
        """
        # Convert BGR to RGB (OpenCV uses BGR by default)
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize the image
        image = self.resize(image)
        
        # Normalize if required
        if self.normalize:
            image = self.normalize_image(image)
        
        return image
    
    def batch_preprocess(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Preprocess a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            Batch of preprocessed images as numpy array
        """
        processed_images = [self.preprocess(img) for img in images]
        return np.array(processed_images)


class ObjectDetector:
    """Class for detecting objects in images using traditional computer vision methods."""
    
    def __init__(self, 
                 method: str = 'contour',
                 blur_kernel_size: int = 5,
                 canny_threshold1: int = 30,
                 canny_threshold2: int = 100,
                 min_contour_area: int = 1000,
                 max_contours: int = 5,
                 background_subtractor: str = 'mog2'):
        """
        Initialize the object detector.
        
        Args:
            method: Detection method ('contour', 'blob', 'background')
            blur_kernel_size: Kernel size for Gaussian blur
            canny_threshold1: First threshold for Canny edge detector
            canny_threshold2: Second threshold for Canny edge detector
            min_contour_area: Minimum contour area to consider
            max_contours: Maximum number of contours to return
            background_subtractor: Background subtraction method ('mog2', 'knn', 'gmg')
        """
        self.method = method
        self.blur_kernel_size = blur_kernel_size
        self.canny_threshold1 = canny_threshold1
        self.canny_threshold2 = canny_threshold2
        self.min_contour_area = min_contour_area
        self.max_contours = max_contours
        
        # Initialize background subtractor if needed
        if self.method == 'background':
            if background_subtractor == 'mog2':
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                    history=500, varThreshold=16, detectShadows=False)
            elif background_subtractor == 'knn':
                self.bg_subtractor = cv2.createBackgroundSubtractorKNN(
                    history=500, dist2Threshold=400.0, detectShadows=False)
            elif background_subtractor == 'gmg':
                self.bg_subtractor = cv2.bgsegm.createBackgroundSubtractorGMG(
                    initializationFrames=120, decisionThreshold=0.8)
            else:
                logger.warning(f"Unknown background subtractor: {background_subtractor}, using MOG2")
                self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()
                
            # Keep a history of frames for background subtraction
            self.frame_history = []
            self.history_size = 30
    
    def detect_by_contour(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects using contour detection.
        
        Args:
            image: Input image
            
        Returns:
            List of detected objects with bounding boxes and contours
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel_size, self.blur_kernel_size), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and get the largest ones
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]
        filtered_contours.sort(key=cv2.contourArea, reverse=True)
        filtered_contours = filtered_contours[:self.max_contours]
        
        # Prepare output
        objects = []
        for contour in filtered_contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            objects.append({
                'bbox': (x, y, w, h),
                'center': (center_x, center_y),
                'contour': contour,
                'area': cv2.contourArea(contour)
            })
        
        return objects
    
    def detect_by_blob(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects using blob detection.
        
        Args:
            image: Input image
            
        Returns:
            List of detected objects with keypoints
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Set up the blob detector parameters
        params = cv2.SimpleBlobDetector_Params()
        
        # Change thresholds
        params.minThreshold = 10
        params.maxThreshold = 200
        
        # Filter by area
        params.filterByArea = True
        params.minArea = 1000
        
        # Filter by circularity
        params.filterByCircularity = False
        
        # Filter by convexity
        params.filterByConvexity = False
        
        # Filter by inertia
        params.filterByInertia = False
        
        # Create detector
        detector = cv2.SimpleBlobDetector_create(params)
        
        # Detect blobs
        keypoints = detector.detect(gray)
        
        # Prepare output
        objects = []
        for kp in keypoints:
            x = int(kp.pt[0] - kp.size // 2)
            y = int(kp.pt[1] - kp.size // 2)
            w = int(kp.size)
            h = int(kp.size)
            
            objects.append({
                'bbox': (x, y, w, h),
                'center': (int(kp.pt[0]), int(kp.pt[1])),
                'size': kp.size,
                'keypoint': kp
            })
        
        return objects
    
    def detect_by_background_subtraction(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects using background subtraction.
        
        Args:
            image: Input image
            
        Returns:
            List of detected objects with bounding boxes
        """
        # Add current frame to history
        self.frame_history.append(image.copy())
        if len(self.frame_history) > self.history_size:
            self.frame_history.pop(0)
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(image)
        
        # Apply morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in the foreground mask
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and get the largest ones
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > self.min_contour_area]
        filtered_contours.sort(key=cv2.contourArea, reverse=True)
        filtered_contours = filtered_contours[:self.max_contours]
        
        # Prepare output
        objects = []
        for contour in filtered_contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate center
            center_x = x + w // 2
            center_y = y + h // 2
            
            objects.append({
                'bbox': (x, y, w, h),
                'center': (center_x, center_y),
                'contour': contour,
                'area': cv2.contourArea(contour),
                'mask': fg_mask[y:y+h, x:x+w]
            })
        
        return objects
    
    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in the image using the selected method.
        
        Args:
            image: Input image
            
        Returns:
            List of detected objects
        """
        if self.method == 'contour':
            return self.detect_by_contour(image)
        elif self.method == 'blob':
            return self.detect_by_blob(image)
        elif self.method == 'background':
            return self.detect_by_background_subtraction(image)
        else:
            logger.error(f"Unknown detection method: {self.method}")
            return []
    
    def crop_objects(self, image: np.ndarray, add_margin: float = 0.1) -> List[np.ndarray]:
        """
        Detect and crop objects from the image.
        
        Args:
            image: Input image
            add_margin: Margin to add around the bounding box (as a fraction of width/height)
            
        Returns:
            List of cropped object images
        """
        objects = self.detect(image)
        cropped_images = []
        
        for obj in objects:
            x, y, w, h = obj['bbox']
            
            # Add margin
            margin_x = int(w * add_margin)
            margin_y = int(h * add_margin)
            
            # Ensure boundaries are within the image
            x1 = max(0, x - margin_x)
            y1 = max(0, y - margin_y)
            x2 = min(image.shape[1], x + w + margin_x)
            y2 = min(image.shape[0], y + h + margin_y)
            
            # Crop the image
            cropped = image[y1:y2, x1:x2]
            cropped_images.append(cropped)
        
        return cropped_images
    
    def visualize_detections(self, image: np.ndarray, objects: List[Dict[str, Any]]) -> np.ndarray:
        """
        Visualize detected objects on the image.
        
        Args:
            image: Input image
            objects: List of detected objects
            
        Returns:
            Image with visualized detections
        """
        vis_image = image.copy()
        
        for i, obj in enumerate(objects):
            # Draw bounding box
            x, y, w, h = obj['bbox']
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw label
            label = f"Object {i+1}"
            cv2.putText(vis_image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw center point
            center_x, center_y = obj['center']
            cv2.circle(vis_image, (center_x, center_y), 5, (0, 0, 255), -1)
            
            # Draw contour if available
            if 'contour' in obj:
                cv2.drawContours(vis_image, [obj['contour']], 0, (255, 0, 0), 2)
        
        return vis_image


class TrashPreprocessingPipeline:
    """Complete preprocessing pipeline for trash classification."""
    
    def __init__(self, 
                 target_size: Tuple[int, int] = (224, 224),
                 detection_method: str = 'contour',
                 normalize: bool = True,
                 batch_size: int = 5):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            target_size: Target image size for model input
            detection_method: Object detection method
            normalize: Whether to normalize pixel values
            batch_size: Batch size for processing
        """
        self.target_size = target_size
        self.batch_size = batch_size
        
        # Initialize object detector
        self.detector = ObjectDetector(method=detection_method)
        
        # Initialize image preprocessor
        self.preprocessor = ImagePreprocessor(target_size=target_size, normalize=normalize)
    
    def process_single_image(self, image: np.ndarray) -> Tuple[List[np.ndarray], List[Dict[str, Any]]]:
        """
        Process a single image through the pipeline.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (processed object images, detected objects)
        """
        # Detect objects
        objects = self.detector.detect(image)
        
        # Crop detected objects
        cropped_objects = self.detector.crop_objects(image)
        
        # Preprocess cropped objects
        processed_objects = [self.preprocessor.preprocess(obj) for obj in cropped_objects]
        
        return processed_objects, objects
    
    def process_batch(self, images: List[np.ndarray]) -> Tuple[np.ndarray, List[List[Dict[str, Any]]]]:
        """
        Process a batch of images through the pipeline.
        
        Args:
            images: List of input images
            
        Returns:
            Tuple of (batch of processed images, list of detected objects per image)
        """
        all_processed_objects = []
        all_objects = []
        
        for image in images:
            processed_objects, objects = self.process_single_image(image)
            all_processed_objects.extend(processed_objects)
            all_objects.append(objects)
        
        # Convert to numpy array if there are any processed objects
        if all_processed_objects:
            all_processed_objects = np.array(all_processed_objects)
        else:
            # Create an empty batch with the correct shape
            all_processed_objects = np.zeros((0, self.target_size[1], self.target_size[0], 3))
        
        return all_processed_objects, all_objects


# Helper functions
def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE) to enhance image contrast.
    
    Args:
        image: Input image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE enhanced image
    """
    # Convert to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE to the L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)
    
    # Merge the CLAHE enhanced L channel with the original A and B channels
    merged_lab = cv2.merge((cl, a, b))
    
    # Convert back to RGB color space
    enhanced_image = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_image


def remove_shadows(image: np.ndarray) -> np.ndarray:
    """
    Remove shadows from the image.
    
    Args:
        image: Input image
        
    Returns:
        Image with shadows removed
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply bilateral filter to smooth the image while preserving edges
    blur = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Apply threshold to get binary image
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Create a mask from the thresholded image
    mask = cv2.bitwise_not(thresh)
    
    # Apply the mask to the original image
    result = image.copy()
    for i in range(3):
        result[:, :, i] = cv2.bitwise_and(result[:, :, i], mask)
    
    return result


def sharpen_image(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0, amount: float = 1.0) -> np.ndarray:
    """
    Sharpen the image using unsharp mask.
    
    Args:
        image: Input image
        kernel_size: Size of Gaussian blur kernel
        sigma: Standard deviation for Gaussian blur
        amount: Strength of sharpening effect
        
    Returns:
        Sharpened image
    """
    # Create a blurred version of the image
    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    # Calculate the unsharp mask
    unsharp_mask = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    
    return unsharp_mask


def auto_adjust_brightness_contrast(image: np.ndarray, clip_hist_percent: float = 1.0) -> np.ndarray:
    """
    Automatically adjust brightness and contrast based on histogram.
    
    Args:
        image: Input image
        clip_hist_percent: Percentage of histogram to clip
        
    Returns:
        Adjusted image
    """
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calculate histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution from histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for i in range(1, hist_size):
        accumulator.append(accumulator[i-1] + float(hist[i]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    # Apply brightness and contrast adjustment
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    
    return adjusted


# Example usage
if __name__ == "__main__":
    # Load a test image
    image_path = "test_image.jpg"
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Initialize the preprocessing pipeline
        pipeline = TrashPreprocessingPipeline(target_size=(224, 224), detection_method='contour')
        
        # Process the image
        processed_objects, objects = pipeline.process_single_image(image)
        
        # Visualize detections
        vis_image = pipeline.detector.visualize_detections(image, objects)
        
        # Display results
        cv2.imshow("Original Image", image)
        cv2.imshow("Detected Objects", vis_image)
        
        if processed_objects:
            # Convert from float back to uint8 for display
            display_obj = (processed_objects[0] * 255).astype(np.uint8)
            cv2.imshow("Processed Object", display_obj)
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}") 