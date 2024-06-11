import os
import cv2
import json
import numpy as np
import re
from sklearn.decomposition import PCA

def get_person_id(image_path):
    # Regex pattern to match 'p{number}'
    pattern = r'p\d+'
    # Find all occurrences of the pattern in the string
    match = re.search(pattern, image_path)
    return match.group() if match else None

def extract_eigenpalm_features(image, pca):
    image_vector = image.flatten()
    eigenpalm_features = pca.transform([image_vector])[0]
    return eigenpalm_features.tolist()

def extract_features(image_path, pca):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    eigenpalm_features = extract_eigenpalm_features(image, pca)
    person_id = get_person_id(image_path)
    return {
        "eigenpalm": eigenpalm_features,
        "person_id": person_id
    }

def get_all_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def train_pca(image_paths, n_components=50):
    images = []
    for file_path in image_paths:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (128, 128))
        images.append(image.flatten())
    images = np.array(images)
    pca = PCA(n_components=n_components)
    pca.fit(images)
    return pca

def main(directory, output_json):
    image_paths = get_all_image_paths(directory)
    pca = train_pca(image_paths)
    features = {}
    for file_path in image_paths:
        features[os.path.basename(file_path)] = extract_features(file_path, pca)
    
    with open(output_json, 'w') as f:
        json.dump(features, f, indent=4)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_eigenpalms.py <directory> <output_json>")
        sys.exit(1)
    directory = sys.argv[1]
    output_json = sys.argv[2]
    main(directory, output_json)
