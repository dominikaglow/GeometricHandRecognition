import os
import sys
import json
import cv2
import numpy as np
import re

def kirsch_kernels():
    return [
        np.array([[5, 5, 5], [-3, 0, -3], [-3, -3, -3]]),  # N
        np.array([[5, 5, -3], [5, 0, -3], [-3, -3, -3]]),  # NE
        np.array([[5, -3, -3], [5, 0, -3], [5, -3, -3]]),  # E
        np.array([[-3, -3, -3], [5, 0, -3], [5, 5, -3]]),  # SE
        np.array([[-3, -3, -3], [-3, 0, -3], [5, 5, 5]]),  # S
        np.array([[-3, -3, -3], [-3, 0, 5], [-3, 5, 5]]),  # SW
        np.array([[-3, -3, 5], [-3, 0, 5], [-3, -3, 5]]),  # W
        np.array([[-3, 5, 5], [-3, 0, 5], [-3, -3, -3]])   # NW
    ]

def calculate_texture_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    texture_code_map = np.zeros_like(gray, dtype=int)
    for i in range(1, gray.shape[0] - 1):
        for j in range(1, gray.shape[1] - 1):
            center = gray[i, j]
            neighbors = [
                gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                gray[i, j-1], gray[i, j+1],
                gray[i+1, j-1], gray[i+1, j], gray[i+1, j+1]
            ]
            differences = np.abs(neighbors - center)
            sorted_indices = np.argsort(-differences)[:2]
            m1, m2 = sorted(sorted_indices)
            texture_code_map[i, j] = m1 * 8 + m2
    return texture_code_map

def calculate_gradient_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gradient_code_map = np.zeros_like(gray, dtype=int)
    max_responses = np.zeros_like(gray, dtype=float)
    kernels = kirsch_kernels()
    for idx, kernel in enumerate(kernels):
        response = cv2.filter2D(gray, -1, kernel)
        update = response > max_responses
        gradient_code_map[update] = idx
        max_responses[update] = response[update]
    return gradient_code_map

def calculate_direction_feature(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gabor_kernels = [cv2.getGaborKernel((21, 21), 4, theta, 10, 0.5, 0, ktype=cv2.CV_32F) for theta in np.linspace(0, np.pi, 12)]
    direction_code_map = np.zeros_like(gray, dtype=int)
    max_responses = np.zeros_like(gray, dtype=float)
    for idx, kernel in enumerate(gabor_kernels):
        response = cv2.filter2D(gray, -1, kernel)
        update = response > max_responses
        direction_code_map[update] = idx
        max_responses[update] = response[update]
    return direction_code_map

def calculate_histograms(feature_map, block_size=16):
    histograms = []
    for i in range(0, feature_map.shape[0], block_size):
        for j in range(0, feature_map.shape[1], block_size):
            block = feature_map[i:i+block_size, j:j+block_size]
            hist, _ = np.histogram(block, bins=np.arange(256), range=(0, 255))
            histograms.append(hist)
    return np.concatenate(histograms)

def get_person_id(image_path):
    # Regex pattern to match 'p{number}'
    pattern = r'p\d+'

    # Find all occurrences of the pattern in the string
    match = re.search(pattern, image_path)

    return match.group() if match else None

def get_all_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def save_features_to_json(image_paths, output_file):
    features_dict = {}
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (128, 128))
        texture = calculate_texture_feature(image)
        gradient = calculate_gradient_feature(image)
        direction = calculate_direction_feature(image)
        hist_texture = calculate_histograms(texture)
        hist_gradient = calculate_histograms(gradient)
        hist_direction = calculate_histograms(direction)
        person_id = get_person_id(image_path)
        features_dict[image_path] = {
            'person_id': person_id,
            'hist_texture': hist_texture.tolist(),
            'hist_gradient': hist_gradient.tolist(),
            'hist_direction': hist_direction.tolist()
        }
    
    with open(output_file, 'w') as f:
        json.dump(features_dict, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_features_to_json.py <directory> <output_file>")
        sys.exit(1)

    directory = sys.argv[1]
    output_file = sys.argv[2]

    image_paths = get_all_image_paths(directory)
    save_features_to_json(image_paths, output_file)

    print(f"Features saved to {output_file}")
