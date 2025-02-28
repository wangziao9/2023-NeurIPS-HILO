# Save predicted percepts and labels
import os
import numpy as np
from src.Acc import evaluate_similarity_ocr

# Load them back to verify
loaded_percepts = np.load('test/predicted_percepts.npy')
loaded_labels = np.load('test/labels.npy')

# Evaluate OCR similarity between predicted percepts and test labels
avg_similarity, similarities, recognized_texts = evaluate_similarity_ocr(loaded_percepts, loaded_labels)

loaded_percepts_opt = np.load('test/predicted_percepts_opt.npy')
loaded_labels_opt = np.load('test/labels_opt.npy')

# Evaluate OCR similarity between predicted percepts and test labels
avg_similarity_opt, similarities_opt, recognized_texts_opt = evaluate_similarity_ocr(loaded_percepts_opt, loaded_labels_opt)

# Print the similarities
print("Individual similarities:")
for i, sim in enumerate(similarities):
    print(f"Image {i}: {sim:.4f}")
print(f"\nAverage similarity: {avg_similarity:.4f}")

print("\nIndividual similarities (optimized):")
for i, sim in enumerate(similarities_opt):
    print(f"Image {i}: {sim:.4f}")
print(f"\nAverage similarity (optimized): {avg_similarity_opt:.4f}")