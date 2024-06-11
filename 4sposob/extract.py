import os
import cv2
import json
import numpy as np
from skimage.feature import local_binary_pattern
import re

def get_person_id(image_path):
    # Regex pattern to match 'p{number}'
    pattern = r'p\d+'
    # Find all occurrences of the pattern in the string
    match = re.search(pattern, image_path)
    return match.group() if match else None

def extract_lbp_features(image, num_points=24, radius=8):
    lbp = local_binary_pattern(image, num_points, radius, method='uniform')
    hist, _ = np.histogram(lbp, bins=np.arange(0, num_points + 3), range=(0, num_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize histogram
    return hist.tolist()

def extract_gabor_features(image, frequencies=(0.1, 0.3, 0.5, 0.7)):
    gabor_features = []
    for frequency in frequencies:
        kernel = cv2.getGaborKernel((21, 21), 8.0, np.pi / 4, frequency, 0.5, 0, ktype=cv2.CV_32F)
        filtered_image = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        hist, _ = np.histogram(filtered_image, bins=64, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)  # Normalize histogram
        gabor_features.append(hist.tolist())
    return gabor_features

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    lbp_features = extract_lbp_features(image)
    gabor_features = extract_gabor_features(image)
    person_id = get_person_id(image_path)
    return {
        "lbp": lbp_features,
        "gabor": gabor_features,
        "person_id": person_id
    }

def get_all_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main(directory, output_json):
    features = {}
    image_paths = get_all_image_paths(directory)
    for file_path in image_paths:
        features[os.path.basename(file_path)] = extract_features(file_path)
    
    with open(output_json, 'w') as f:
        json.dump(features, f, indent=4)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_features.py <directory> <output_json>")
        sys.exit(1)
    directory = sys.argv[1]
    output_json = sys.argv[2]
    main(directory, output_json)
