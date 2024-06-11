import os
import cv2
import json
import numpy as np
from sklearn.cluster import KMeans
import re

def get_person_id(image_path):
    # Regex pattern to match 'p{number}'
    pattern = r'p\d+'
    # Find all occurrences of the pattern in the string
    match = re.search(pattern, image_path)
    return match.group() if match else None

def extract_sift_features(image, n_clusters=50):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    if descriptors is None:
        return np.zeros(n_clusters).tolist()  # Handle case with no keypoints
    n_descriptors = len(descriptors)
    if n_descriptors < n_clusters:
        # Pad descriptors with zeros
        padded_descriptors = np.vstack([descriptors, np.zeros((n_clusters - n_descriptors, descriptors.shape[1]))])
        descriptors = padded_descriptors
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(descriptors)
    hist, _ = np.histogram(kmeans.labels_, bins=np.arange(0, n_clusters + 1))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize histogram
    return hist.tolist()

def extract_edge_histogram(image):
    edges = cv2.Canny(image, 100, 200)
    hist, _ = np.histogram(edges, bins=64, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize histogram
    return hist.tolist()

def extract_features(image_path, n_clusters=50):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    
    sift_features = extract_sift_features(image, n_clusters=n_clusters)
    edge_histogram = extract_edge_histogram(image)
    person_id = get_person_id(image_path)
    
    return {
        "sift": sift_features,
        "edge_histogram": edge_histogram,
        "person_id": person_id
    }

def get_all_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def main(directory, output_json, n_clusters=50):
    features = {}
    image_paths = get_all_image_paths(directory)
    for file_path in image_paths:
        features[os.path.basename(file_path)] = extract_features(file_path, n_clusters=n_clusters)
    
    with open(output_json, 'w') as f:
        json.dump(features, f, indent=4)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python extract_new_features.py <directory> <output_json>")
        sys.exit(1)
    directory = sys.argv[1]
    output_json = sys.argv[2]
    main(directory, output_json)
