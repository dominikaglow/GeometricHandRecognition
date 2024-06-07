import os
import sys
import cv2
import numpy as np
from feature_extraction import calculate_texture_feature, calculate_gradient_feature, calculate_direction_feature, calculate_histograms, match_palmprints

def load_images_from_folder(folder):
    images = []
    filenames = []
    # Walk through all directories and subdirectories
    for root, dirs, files in os.walk(folder):
        for file in sorted(files):
            if file.endswith('.jpg') or file.endswith('.jpeg'):  # Specify the file format
                path = os.path.join(root, file)
                img = cv2.imread(path)
                if img is not None:
                    img = cv2.resize(img, (128, 128))  # Resize images
                    images.append(img)
                    filenames.append(os.path.relpath(path, start=folder))  # Save relative path for clarity
    return images, filenames

def compare_images(images):
    n = len(images)
    scores = {}
    histograms = []

    # Calculate histograms for all images first
    for img in images:
        texture = calculate_texture_feature(img)
        gradient = calculate_gradient_feature(img)
        direction = calculate_direction_feature(img)
        hist_texture = calculate_histograms(texture)
        hist_gradient = calculate_histograms(gradient)
        hist_direction = calculate_histograms(direction)
        histograms.append([hist_texture, hist_gradient, hist_direction])

    # Compare every image against every other image
    for i in range(n):
        scores[i] = []
        for j in range(n):
            if i != j:
                weights = [0.1, 0.1, 0.8]  # Example weights
                score = match_palmprints(histograms[i], histograms[j], weights)
                scores[i].append((j, score))

    return scores

def print_comparison_results(filenames, scores):
    for i in range(len(filenames)):
        print(filenames[i])
        sorted_scores = sorted(scores[i], key=lambda x: x[1])  # Sort based on the score
        for comparison in sorted_scores:
            print(f"{filenames[comparison[0]]} {comparison[1]:.2f}")
        print()  # Adds a newline for better readability

def main():
    if len(sys.argv) < 2:
        print("Usage: python compare_images.py <folder_path>")
        return

    folder_path = sys.argv[1]
    images, filenames = load_images_from_folder(folder_path)
    scores = compare_images(images)
    print_comparison_results(filenames, scores)

if __name__ == "__main__":
    main()
