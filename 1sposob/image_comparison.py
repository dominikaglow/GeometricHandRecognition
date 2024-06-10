import os
import sys
import json
import numpy as np

def chi_square_distance(hist1, hist2):
    return np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))

def match_palmprints(hist1, hist2, weights):
    scores = [chi_square_distance(h1, h2) for h1, h2 in zip(hist1, hist2)]
    return np.dot(weights, scores)

def load_features(json_file):
    with open(json_file, 'r') as f:
        features_dict = json.load(f)
    return features_dict

def check_classification(features_dict):
    correct_count = 0
    total_count = len(features_dict)
    
    for image_path1, data1 in features_dict.items():
        scores = []
        hist1 = [np.array(data1['hist_texture']), np.array(data1['hist_gradient']), np.array(data1['hist_direction'])]
        person_id1 = data1['person_id']
        
        for image_path2, data2 in features_dict.items():
            if image_path1 != image_path2:
                hist2 = [np.array(data2['hist_texture']), np.array(data2['hist_gradient']), np.array(data2['hist_direction'])]
                score = match_palmprints(hist1, hist2, [0.1, 0.1, 0.8])
                scores.append((image_path2, score))
        
        scores.sort(key=lambda x: x[1])  # Sort scores in ascending order
        closest_match = scores[0][0]
        person_id2 = features_dict[closest_match]['person_id']
        
        if person_id1 == person_id2:
            correct_count += 1
        
        accuracy = (correct_count / (list(features_dict.keys()).index(image_path1) + 1)) * 100
        print(f"Correctly classified: {correct_count}/{list(features_dict.keys()).index(image_path1) + 1} ({accuracy:.2f}%)")
    
    return correct_count, total_count

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_features_from_json.py <features_json_file>")
        sys.exit(1)

    json_file = sys.argv[1]

    features_dict = load_features(json_file)
    correct_count, total_count = check_classification(features_dict)
    accuracy = (correct_count / total_count) * 100
    print(f"Final Accuracy: {correct_count}/{total_count} ({accuracy:.2f}%)")
