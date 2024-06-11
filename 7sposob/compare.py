import json
import numpy as np
import sys

def euclidean_distance(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))

def compare_features(features1, features2):
    distance_pca = euclidean_distance(features1["pca"], features2["pca"])
    distance_fft = euclidean_distance(features1["fft"], features2["fft"])
    return (distance_pca + distance_fft) / 2

def find_most_similar_images(features_data):
    image_names = list(features_data.keys())
    correct_matches = 0
    total_images = len(image_names)
    
    for i in range(total_images):
        min_score = float('inf')
        most_similar_image = None
        original_image_name = image_names[i]
        original_features = features_data[original_image_name]
        original_person_id = original_features["person_id"]

        for j in range(total_images):
            if i == j:
                continue
            comparison_image_name = image_names[j]
            comparison_features = features_data[comparison_image_name]
            score = compare_features(original_features, comparison_features)
            
            if score < min_score:
                min_score = score
                most_similar_image = comparison_image_name
        
        if most_similar_image:
            most_similar_person_id = features_data[most_similar_image]["person_id"]
            if original_person_id == most_similar_person_id:
                correct_matches += 1
        
        print("Correctly classified: {}/{}".format(correct_matches, i + 1))

    accuracy_rate = correct_matches / total_images
    print(f"Accuracy rate: {accuracy_rate:.2%}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python compare_features_pca_fft.py <features_json>")
        sys.exit(1)
    
    features_json = sys.argv[1]
    
    with open(features_json, 'r') as f:
        features_data = json.load(f)
    
    find_most_similar_images(features_data)
