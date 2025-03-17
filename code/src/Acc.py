"""
Acc.py - Chinese Character Recognition Accuracy Measurement

This module implements functions to measure the recognition accuracy of predicted percepts 
from the HILO optimization on Chinese character datasets.
"""

import numpy as np
import tensorflow as tf
import os
import pickle
from tqdm import tqdm
import cv2
import Levenshtein
from paddleocr import PaddleOCR


# Calculate similarity using PaddleOCR and Levenshtein distance
def calculate_similarity_ocr(image, label):
    """
    Calculate similarity between an image and a label using PaddleOCR and Levenshtein distance
    
    Parameters
    ----------
    image : numpy.ndarray
        Image containing the character
    label : str or int
        True label for the character
        
    Returns
    -------
    similarity : float
        Similarity score between 0 and 1
    recognized_text : str
        Text recognized by PaddleOCR
    """
    # Initialize PaddleOCR
    ocr = PaddleOCR(lang='ch', det_model_dir='ch_ppocr_server')
    
    # Ensure the image is properly formatted for PaddleOCR
    if len(image.shape) == 2:
        # Convert grayscale to RGB
        image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    elif len(image.shape) == 3 and image.shape[2] == 1:
        # Convert single channel to RGB
        image = cv2.cvtColor(image.squeeze().astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
    # Scale image values to 0-255 if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Run OCR
    results = ocr.ocr(image, det=False, cls=False)
    
    # Extract recognized text
    recognized_text = results[0][0][0]
    
    # Convert label to string if it's not already
    if isinstance(label, int):
        try:
            label_str = chr(label)
        except:
            label_str = str(label)
    else:
        label_str = str(label)
    
    # Calculate Levenshtein distance
    if recognized_text:
        distance = Levenshtein.distance(recognized_text, label_str)
        max_len = max(len(recognized_text), len(label_str))
        similarity = 1.0 - (distance / max_len) if max_len > 0 else 0.0
    else:
        similarity = 0.0
    
    return similarity, recognized_text

def evaluate_similarity_ocr(percepts, labels):
    """
    Evaluate similarity between percepts and labels using PaddleOCR and Levenshtein distance
    
    Parameters
    ----------
    percepts : numpy.ndarray
        Percepts to evaluate
    labels : numpy.ndarray
        True labels for the percepts
        
    Returns
    -------
    avg_similarity : float
        Average similarity score
    similarities : list
        List of individual similarity scores
    recognized_texts : list
        List of recognized texts
    """
    similarities = []
    recognized_texts = []
    
    for i in tqdm(range(len(percepts)), desc="OCR Recognition"):
        similarity, recognized_text = calculate_similarity_ocr(percepts[i], labels[i])
        similarities.append(similarity)
        recognized_texts.append(recognized_text)
    
    avg_similarity = np.mean(similarities)
    return avg_similarity, similarities, recognized_texts