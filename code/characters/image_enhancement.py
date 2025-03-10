"""
Image Enhancement for CASIA character dataset

This module provides functions to preprocess and enhance character images:
1. Dilation - Thickens character strokes
2. Downsampling - Reduces image size while preserving essential features
3. Sharpening and Edge Enhancement - Increases contrast at edges

These methods help improve OCR recognition accuracy.
"""

import numpy as np
import cv2

def dilate_image(image, kernel_size=3, iterations=1):
    """
    Apply dilation to thicken character strokes.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale)
    kernel_size : int
        Size of the dilation kernel
    iterations : int
        Number of times to apply dilation
        
    Returns
    -------
    numpy.ndarray
        Dilated image
    """
    # Ensure image is in correct format
    if image.max() <= 1.0:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)
    
    # Create kernel for dilation
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    
    # Apply dilation
    dilated = cv2.dilate(img, kernel, iterations=iterations)
    
    # Normalize to match input format
    if image.max() <= 1.0:
        return dilated.astype(float) / 255.0
    return dilated

def downsample_image(image, scale_factor=0.5, interpolation=cv2.INTER_AREA):
    """
    Downsample image to reduce noise and size.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale)
    scale_factor : float
        Scale factor for downsampling (0.5 means reduce to half size)
    interpolation : int
        Interpolation method (default is INTER_AREA which is best for downsampling)
        
    Returns
    -------
    numpy.ndarray
        Downsampled image
    """
    # Calculate new dimensions
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
    
    # Ensure image is in correct format
    if image.max() <= 1.0:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)
    
    # Downsample
    downsampled = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    # Upsample back to original size to maintain compatibility
    upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to match input format
    if image.max() <= 1.0:
        return upsampled.astype(float) / 255.0
    return upsampled

def sharpen_image(image, kernel_size=3, alpha=1.5):
    """
    Sharpen image to enhance edges.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale)
    kernel_size : int
        Size of the Gaussian blur kernel
    alpha : float
        Sharpening strength
        
    Returns
    -------
    numpy.ndarray
        Sharpened image
    """
    # Ensure image is in correct format
    if image.max() <= 1.0:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Calculate unsharp mask (original - blurred)
    unsharp_mask = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)
    
    # Normalize to match input format
    if image.max() <= 1.0:
        return unsharp_mask.astype(float) / 255.0
    return unsharp_mask

def enhance_edges(image, low_threshold=50, high_threshold=150):
    """
    Enhance edges in the image using Canny edge detection.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale)
    low_threshold : int
        Lower threshold for the Canny edge detector
    high_threshold : int
        Higher threshold for the Canny edge detector
        
    Returns
    -------
    numpy.ndarray
        Edge-enhanced image
    """
    # Ensure image is in correct format
    if image.max() <= 1.0:
        img = (image * 255).astype(np.uint8)
    else:
        img = image.astype(np.uint8)
    
    # Detect edges
    edges = cv2.Canny(img, low_threshold, high_threshold)
    
    # Blend original with edges
    enhanced = cv2.addWeighted(img, 0.7, edges, 0.3, 0)
    
    # Normalize to match input format
    if image.max() <= 1.0:
        return enhanced.astype(float) / 255.0
    return enhanced

def apply_all_enhancements(image, params=None):
    """
    Apply all enhancement methods in sequence.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale)
    params : dict
        Dictionary of parameters for each enhancement method
        
    Returns
    -------
    numpy.ndarray
        Enhanced image
    """
    if params is None:
        params = {
            'dilate': {'kernel_size': 3, 'iterations': 1},
            'downsample': {'scale_factor': 0.5},
            'sharpen': {'kernel_size': 3, 'alpha': 1.5},
            'edge': {'low_threshold': 50, 'high_threshold': 150}
        }
    
    # Apply dilation
    enhanced = dilate_image(image, **params['dilate'])
    
    # Apply downsampling
    enhanced = downsample_image(enhanced, **params['downsample'])
    
    # Apply sharpening
    enhanced = sharpen_image(enhanced, **params['sharpen'])
    
    # Apply edge enhancement
    enhanced = enhance_edges(enhanced, **params['edge'])
    
    return enhanced