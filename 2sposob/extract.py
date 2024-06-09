import os
import sys
import json
import cv2
import numpy as np
from skimage.feature import hog
import re

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image at path {image_path} cannot be read.")
    image = cv2.resize(image, (128, 128))
    image = cv2.GaussianBlur(image, (5, 5), 0)
    return image

def extract_features(image):
    features, _ = hog(image, pixels_per_cell=(16, 16), cells_per_block=(2, 2), visualize=True)
    return features.tolist()  # Convert features to list for JSON serialization

def get_all_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def get_person_id(image_path):
    # Regex pattern to match 'p{number}'
    pattern = r'p\d+'

    # Find all occurrences of the pattern in the string
    match = re.search(pattern, image_path)

    return match.group() if match else None

def save_features_to_json(image_paths, output_file):
    features_dict = {}
    for image_path in image_paths:
        image = preprocess_image(image_path)
        features = extract_features(image)
        person_id = get_person_id(image_path)
        features_dict[image_path] = {
            'person_id': person_id,
            'features': features
        }
    
    with open(output_file, 'w') as f:
        json.dump(features_dict, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script_name.py <directory> <output_file>")
        sys.exit(1)

    directory = sys.argv[1]
    output_file = sys.argv[2]

    image_paths = get_all_image_paths(directory)
    save_features_to_json(image_paths, output_file)

    print(f"Features saved to {output_file}")
