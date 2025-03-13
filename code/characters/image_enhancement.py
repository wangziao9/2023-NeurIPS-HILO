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
import matplotlib.pyplot as plt

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
    img = (image * 255).astype(np.uint8)
    
    # Create kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    # Apply dilation
    dilated =  cv2.dilate(img, kernel, iterations=iterations)
    
    # Normalize to match input format
    dilated = np.expand_dims(dilated.astype(float) / 255.0, axis=-1)
    return dilated

def erode_image(image, kernel_size=3, iterations=1):
    """
    Apply erode to thin character strokes.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image (grayscale)
    kernel_size : int
        Size of the erode kernel
    iterations : int
        Number of times to apply erode
        
    Returns
    -------
    numpy.ndarray
        Dilated image
    """
    # Ensure image is in correct format
    img = (image * 255).astype(np.uint8)
    
    # Create kernel for dilation
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    
    # Apply dilation
    eroded = cv2.erode(img, kernel, iterations=iterations)
    
    # Normalize to match input format
    eroded = np.expand_dims(eroded.astype(float) / 255.0, axis=-1)
    return eroded

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
    img = (image * 255).astype(np.uint8)
    
    # Downsample
    downsampled = cv2.resize(img, (new_w, new_h), interpolation=interpolation)
    
    # Upsample back to original size to maintain compatibility
    upsampled = cv2.resize(downsampled, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Normalize to match input format
    upsampled =  np.expand_dims(upsampled.astype(float) / 255.0, axis=-1)
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
    img = (image * 255).astype(np.uint8)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    # Calculate unsharp mask (original - blurred)
    unsharp_mask = cv2.addWeighted(img, 1 + alpha, blurred, -alpha, 0)
    
    # Normalize to match input format
    unsharp_mask = np.expand_dims(unsharp_mask.astype(float) / 255.0, axis=-1)
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
    img = (image * 255).astype(np.uint8)
    
    # Detect edges
    edges = cv2.Canny(img, low_threshold, high_threshold)
    
    # Blend original with edges
    enhanced = cv2.addWeighted(img, 0.7, edges, 0.3, 0)
    
    # Normalize to match input format
    enhanced = np.expand_dims(enhanced.astype(float) / 255.0, axis=-1)
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

    # img = (image * 255).astype(np.uint8)
    # _, binary_input = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    # img = np.expand_dims(binary_input.astype(float) / 255.0, axis=-1)
    
    # Apply dilation
    enhanced = dilate_image(image, **params['dilate'])
    
    # Apply downsampling
    enhanced = downsample_image(enhanced, **params['downsample'])
    
    # Apply sharpening
    enhanced = sharpen_image(enhanced, **params['sharpen'])
    
    # Apply edge enhancement
    enhanced = enhance_edges(enhanced, **params['edge'])
    enhanced = erode_image(enhanced, **params['erode'])
    
    return enhanced