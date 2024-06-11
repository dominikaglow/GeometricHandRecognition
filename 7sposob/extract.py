import os
import cv2
import json
import numpy as np
from sklearn.decomposition import PCA
import re

def get_person_id(image_path):
    pattern = r'p\d+'
    match = re.search(pattern, image_path)
    return match.group() if match else None

def extract_pca_features(image, num_components=0.5):
    h, w = image.shape
    reshaped_image = image.reshape(-1, h * w)  # Reshape to a 1D array for PCA
    pca = PCA(n_components=num_components)
    pca.fit(reshaped_image)
    pca_features = pca.transform(reshaped_image)
    return pca_features.flatten().tolist()

def extract_fft_features(image):
    f_transform = np.fft.fft2(image)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-10)
    central_part = magnitude_spectrum[64-16:64+16, 64-16:64+16]  # Assuming 128x128 resized images
    return central_part.flatten().tolist()

def extract_features(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (128, 128))
    
    pca_features = extract_pca_features(image)
    fft_features = extract_fft_features(image)
    person_id = get_person_id(image_path)
    
    return {
        "pca": pca_features,
        "fft": fft_features,
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
        print("Usage: python extract_features_pca_fft.py <directory> <output_json>")
        sys.exit(1)
    directory = sys.argv[1]
    output_json = sys.argv[2]
    main(directory, output_json)
