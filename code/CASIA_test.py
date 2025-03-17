# Save predicted percepts and labels
import os
import numpy as np
from src.Acc import evaluate_similarity_ocr
import matplotlib.pyplot as plt

# Load them back to verify
loaded_percepts_target = np.load('code/test/origin.npy')
loaded_labels_target = np.load('code/test/labels_origin.npy')

# Evaluate OCR similarity between predicted percepts and test labels
avg_similarity_target, similarities_target, recognized_texts_target = evaluate_similarity_ocr(loaded_percepts_target, loaded_labels_target)


# Load them back to verify
loaded_percepts = np.load('code/test/predicted_percepts.npy')
loaded_labels = np.load('code/test/labels.npy')

# Evaluate OCR similarity between predicted percepts and test labels
avg_similarity, similarities, recognized_texts = evaluate_similarity_ocr(loaded_percepts, loaded_labels)

loaded_percepts_opt = np.load('code/test/predicted_percepts_opt.npy')
loaded_labels_opt = np.load('code/test/labels_opt.npy')

# Evaluate OCR similarity between predicted percepts and test labels
avg_similarity_opt, similarities_opt, recognized_texts_opt = evaluate_similarity_ocr(loaded_percepts_opt, loaded_labels_opt)

# Print the similarities
print("Individual similarities (origin):")
for i, sim in enumerate(similarities_target):
    print(f"Image {i}: {sim:.4f}")
print(f"\nAverage similarity (origin): {avg_similarity_target:.4f}")
print(f"{recognized_texts_target}")

print("Individual similarities:")
for i, sim in enumerate(similarities):
    print(f"Image {i}: {sim:.4f}")
print(f"\nAverage similarity: {avg_similarity:.4f}")
print(f"{recognized_texts}")

print("\nIndividual similarities (optimized):")
for i, sim in enumerate(similarities_opt):
    print(f"Image {i}: {sim:.4f}")
print(f"\nAverage similarity (optimized): {avg_similarity_opt:.4f}")
print(f"{recognized_texts_opt}")


fig, axes = plt.subplots(3, 10, figsize=(10*3, 6))
for idx, (target, pred, pred_opt) in enumerate(zip(loaded_percepts_target, loaded_percepts, loaded_percepts_opt)): 
    axes[0, idx].imshow(target, cmap='gray')
    axes[1, idx].imshow(pred, cmap='gray')
    axes[2, idx].imshow(pred_opt, cmap='gray')
# make it look nice
for ax in axes.ravel():
    ax.axis('off')
fig.tight_layout()
plt.show()