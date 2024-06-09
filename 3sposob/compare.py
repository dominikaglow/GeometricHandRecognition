import os
import sys
import json
import numpy as np

def load_features(json_file):
    with open(json_file, 'r') as f:
        features_dict = json.load(f)
    return features_dict

def compare_features(features1, features2):
    if features1 is None or features2 is None:
        return float('inf')  # Return a large value if features are missing

    # Pad the shorter list with zeros
    max_length = max(len(features1), len(features2))
    features1 = np.pad(features1, (0, max_length - len(features1)), 'constant')
    features2 = np.pad(features2, (0, max_length - len(features2)), 'constant')

    distance = np.linalg.norm(features1 - features2)
    return distance

def check_classification(features_dict):
    correct_count = 0
    total_count = len(features_dict)
    
    for image_path1, data1 in features_dict.items():
        scores = []
        features1 = data1['features']
        person_id1 = data1['person_id']
        
        for image_path2, data2 in features_dict.items():
            if image_path1 != image_path2:
                features2 = data2['features']
                distance = compare_features(features1, features2)
                scores.append((image_path2, distance))
        
        scores.sort(key=lambda x: x[1])  # Sort scores in ascending order
        closest_match = scores[0][0]
        person_id2 = features_dict[closest_match]['person_id']
        print(f"Closest match for {person_id1}: {person_id2}")
        if person_id1 == person_id2:
            correct_count += 1
        
        accuracy = (correct_count / (list(features_dict.keys()).index(image_path1) + 1)) * 100
        print(f"Correctly classified: {correct_count}/{list(features_dict.keys()).index(image_path1) + 1} ({accuracy:.2f}%)")
    
    return correct_count, total_count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <features_json_file>")
        sys.exit(1)

    json_file = sys.argv[1]

    features_dict = load_features(json_file)
    correct_count, total_count = check_classification(features_dict)
    accuracy = (correct_count / total_count) * 100
    print(f"Final Accuracy: {correct_count}/{total_count} ({accuracy:.2f}%)")
