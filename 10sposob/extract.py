import os
import cv2
import json
import numpy as np
import re
import matplotlib.pyplot as plt

def get_person_id(image_path):
    # Regex pattern to match 'p{number}'
    pattern = r'p\d+'
    # Find all occurrences of the pattern in the string
    match = re.search(pattern, image_path)
    return match.group() if match else None

def normalize_hand_orientation(image):
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour which should be the hand
    hand_contour = max(contours, key=cv2.contourArea)
    
    # Calculate the bounding rectangle and minimum area rectangle
    rect = cv2.minAreaRect(hand_contour)
    box = cv2.boxPoints(rect)
    box = np.array(box, dtype=np.intp)  # Updated to avoid the deprecation warning
    
    # Get the angle and size of the hand
    angle = rect[-1]
    if angle < -45:
        angle = 90 + angle

    # Rotate the image to normalize the orientation
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    # Find the new contour after rotation
    gray_rotated = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    _, thresh_rotated = cv2.threshold(gray_rotated, 127, 255, cv2.THRESH_BINARY)
    contours_rotated, _ = cv2.findContours(thresh_rotated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour which should be the hand
    hand_contour_rotated = max(contours_rotated, key=cv2.contourArea)
    
    # Crop the rotated image around the hand contour
    x, y, w, h = cv2.boundingRect(hand_contour_rotated)
    hand_roi = rotated[y:y+h, x:x+w]
    
    # Resize to a fixed size
    hand_roi_resized = cv2.resize(hand_roi, (300, 300))
    
    return hand_roi_resized

def extract_hand_geometry_features(image, num_fingers=5):
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour which should be the hand
    hand_contour = max(contours, key=cv2.contourArea)
    
    # Find the convex hull and defects
    hull = cv2.convexHull(hand_contour, returnPoints=False)
    hull[::-1].sort(axis=0)
    defects = cv2.convexityDefects(hand_contour, hull)
    
    # Initialize lists to hold lengths and widths
    finger_lengths = []
    finger_widths = []
    
    if defects is not None:
        for i in range(defects.shape[0]):
            s, e, f, _ = defects[i, 0]
            start = tuple(hand_contour[s][0])
            end = tuple(hand_contour[e][0])
            far = tuple(hand_contour[f][0])
            finger_lengths.append(np.linalg.norm(np.array(start) - np.array(end)))
            finger_widths.append(np.linalg.norm(np.array(start) - np.array(far)))
            
            # Draw lines on the image for visualization
            cv2.line(image, start, end, (0, 255, 0), 2)
            cv2.line(image, start, far, (255, 0, 0), 2)

    # Ensure the lengths and widths vectors are of the same size
    if len(finger_lengths) < num_fingers:
        finger_lengths.extend([0] * (num_fingers - len(finger_lengths)))
    if len(finger_widths) < num_fingers:
        finger_widths.extend([0] * (num_fingers - len(finger_widths)))
    
    # Trim the lists to the fixed size
    finger_lengths = finger_lengths[:num_fingers]
    finger_widths = finger_widths[:num_fingers]

    return finger_lengths + finger_widths

def extract_palmprint_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    
    # Extract texture features (e.g., using Local Binary Pattern)
    lbp = cv2.Laplacian(gray, cv2.CV_64F)
    hist, _ = np.histogram(lbp, bins=64, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # Normalize histogram
    
    return hist.tolist()

def extract_features(image_path):
    image = cv2.imread(image_path)
    person_id = get_person_id(image_path)
    
    # Normalize the hand orientation and extract the hand ROI
    hand_roi = normalize_hand_orientation(image)
    
    hand_geometry_features = extract_hand_geometry_features(hand_roi)
    palmprint_features = extract_palmprint_features(hand_roi)
    
    return {
        "hand_geometry": hand_geometry_features,
        "palmprint": palmprint_features,
        "person_id": person_id,
        "hand_roi": hand_roi  # Added for visualization
    }

def get_all_image_paths(directory):
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                image_paths.append(os.path.join(root, file))
    return image_paths

def save_images(images, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, img in enumerate(images):
        cv2.imwrite(os.path.join(output_dir, f'hand_roi_{i+1}.png'), img)

def main(directory, output_json, output_images_dir):
    features = {}
    image_paths = get_all_image_paths(directory)
    hand_rois = []
    for i, file_path in enumerate(image_paths):
        feature = extract_features(file_path)
        # Remove hand_roi from features to avoid JSON serialization error
        hand_roi = feature.pop("hand_roi")
        features[os.path.basename(file_path)] = feature
        if i < 10:
            hand_rois.append(hand_roi)
    
    with open(output_json, 'w') as f:
        json.dump(features, f, indent=4)

    save_images(hand_rois, output_images_dir)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python extract_hand_palm_features.py <directory> <output_json> <output_images_dir>")
        sys.exit(1)
    directory = sys.argv[1]
    output_json = sys.argv[2]
    output_images_dir = sys.argv[3]
    main(directory, output_json, output_images_dir)
