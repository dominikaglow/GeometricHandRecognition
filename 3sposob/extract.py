import os
import sys
import json
import cv2
import numpy as np

def capture_image(image_path):
    image = cv2.imread(image_path)
    return image

def get_binary_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def get_contour(binary_image):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        return max(contours, key=cv2.contourArea)
    return None

def extract_convex_hull(contour):
    hull = cv2.convexHull(contour, returnPoints=False)
    return hull

def simplify_contour(contour):
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

def extract_convexity_defects(contour, hull):
    if len(hull) > 3:  # Ensure there are enough points to form defects
        defects = cv2.convexityDefects(contour, hull)
        return defects
    return None

def extract_fingertips(contour, hull):
    hull_points = [tuple(contour[idx][0]) for idx in hull[:, 0]]
    return hull_points

def extract_finger_knuckles(contour, defects):
    knuckles = []
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, _ = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            knuckle = ((start[0] + far[0]) // 2, (start[1] + far[1]) // 2)
            knuckles.append(knuckle)
    return knuckles

def extract_distances(fingertips, knuckles):
    distances = []
    for tip in fingertips:
        for knuckle in knuckles:
            distance = np.linalg.norm(np.array(tip) - np.array(knuckle))
            distances.append(distance)
    return distances

def extract_features(image_path):
    image = capture_image(image_path)
    binary_image = get_binary_image(image)
    contour = get_contour(binary_image)

    if contour is not None:
        contour = simplify_contour(contour)  # Simplify the contour
        hull = extract_convex_hull(contour)
        defects = extract_convexity_defects(contour, hull)
        fingertips = extract_fingertips(contour, hull)
        knuckles = extract_finger_knuckles(contour, defects)
        distances = extract_distances(fingertips, knuckles)
        return distances
    else:
        return None

def get_person_id(image_path):
    return os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(image_path))))

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
        features = extract_features(image_path)
        if features is not None:
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
