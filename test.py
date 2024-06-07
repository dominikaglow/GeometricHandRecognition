import cv2
import numpy as np
from scipy.spatial import distance

# Function to preprocess the palm print image (without rotation)
def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply GaussianBlur to reduce noise and improve feature detection
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply binary thresholding
    _, thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)
    
    return thresh

# Function to extract geometric features from the palm print
def extract_features(image):
    # Find contours
    contours, _ = cv2.findContours(image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)
    
    # Extract geometric features
    features = cv2.HuMoments(cv2.moments(contour)).flatten()
    
    return features

# Function to calculate similarity between two palm prints
def calculate_similarity(image1, image2):
    # Preprocess images
    img1 = preprocess_image(image1)
    img2 = preprocess_image(image2)
    
    # Extract features
    features1 = extract_features(img1)
    features2 = extract_features(img2)
    
    # Calculate similarity (using Euclidean distance here, but other metrics can be used)
    similarity = distance.euclidean(features1, features2)
    
    return similarity

# Load palm print images
image1 = cv2.imread('Example_data/p1/Hand/Left/p1_l_2.jpg')
image2 = cv2.imread('Example_data/p1/Hand/Left/p1_l_3.jpg')

# Calculate similarity
similarity = calculate_similarity(image1, image2)

print(f"Similarity score: {similarity}")
