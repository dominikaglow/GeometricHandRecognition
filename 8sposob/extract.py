import os
import cv2
import json
import numpy as np
from skimage.feature import local_binary_pattern, hog
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

def extract_hog_features(image):
    hog_features, _ = hog(image, orientations=8, pixels_per_cell=(16, 16), 
                          cells_per_block=(1, 1), visualize=True)
    return hog_features.tolist()

def extract_color_histogram(image):
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist.tolist()

def extract_features(image_path):
    image = cv2.imread(image_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image, (128, 128))
    image = cv2.resize(image, (128, 128))
    
    lbp_features = extract_lbp_features(gray_image)
    hog_features = extract_hog_features(gray_image)
    color_histogram = extract_color_histogram(image)
    person_id = get_person_id(image_path)
    
    return {
        "lbp": lbp_features,
        "hog": hog_features,
        "color_histogram": color_histogram,
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
        print("Usage: python extract_combined.py <directory> <output_json>")
        sys.exit(1)
    directory = sys.argv[1]
    output_json = sys.argv[2]
    main(directory, output_json)
