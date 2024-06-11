import json
import numpy as np
import sys

def chi_square_distance(hist1, hist2):
    """
    Calculate the Chi-square distance between two histograms.
    """
    hist1 = np.array(hist1)
    hist2 = np.array(hist2)
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-6))

def compare_features(features1, features2):
    """
    Compare the SIFT and Edge Histogram features of two images and return their combined distance.
    """
    chi_sift = chi_square_distance(features1["sift"], features2["sift"])
    chi_edge_histogram = chi_square_distance(features1["edge_histogram"], features2["edge_histogram"])
    return 0.5 * chi_sift + 0.5 * chi_edge_histogram

def find_most_similar_images(features_data):
    """
    Find the most similar image for each image in the dataset and calculate the accuracy rate based on person ID.
    """
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
        print("Usage: python compare_new_features.py <features_json>")
        sys.exit(1)
    
    features_json = sys.argv[1]
    
    with open(features_json, 'r') as f:
        features_data = json.load(f)
    
    find_most_similar_images(features_data)
