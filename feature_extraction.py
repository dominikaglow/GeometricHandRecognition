import cv2
import numpy as np
import sys

def kirsch_kernels():
    # Define all 8 Kirsch kernels for edge detection
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
    # Assume the center point and its eight neighbors
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
    # Split feature map into blocks and calculate histograms
    histograms = []
    for i in range(0, feature_map.shape[0], block_size):
        for j in range(0, feature_map.shape[1], block_size):
            block = feature_map[i:i+block_size, j:j+block_size]
            hist, _ = np.histogram(block, bins=np.arange(256), range=(0, 255))
            histograms.append(hist)
    return np.concatenate(histograms)

def chi_square_distance(hist1, hist2):
    return np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))

def match_palmprints(hist1, hist2, weights):
    scores = [chi_square_distance(h1, h2) for h1, h2 in zip(hist1, hist2)]
    return np.dot(weights, scores)

def main():
    
    if len(sys.argv) < 3:
        print("Usage: python test.py <image1> <image2>")
        return
    
    # Load and preprocess palmprint images
    image1 = cv2.imread(sys.argv[1])
    image2 = cv2.imread(sys.argv[2])

    # Resize images to 128x128 pixels
    image1 = cv2.resize(image1, (128, 128))
    image2 = cv2.resize(image2, (128, 128))

    # Calculate feature maps
    texture1, gradient1, direction1 = calculate_texture_feature(image1), calculate_gradient_feature(image1), calculate_direction_feature(image1)
    texture2, gradient2, direction2 = calculate_texture_feature(image2), calculate_gradient_feature(image2), calculate_direction_feature(image2)

    # Calculate histograms for each type of feature
    hist_texture1 = calculate_histograms(texture1)
    hist_gradient1 = calculate_histograms(gradient1)
    hist_direction1 = calculate_histograms(direction1)
    hist_texture2 = calculate_histograms(texture2)
    hist_gradient2 = calculate_histograms(gradient2)
    hist_direction2 = calculate_histograms(direction2)

    # Matching and fusion of histograms
    weights = [0.1, 0.1, 0.8]  # Example weights
    score = match_palmprints([hist_texture1, hist_gradient1, hist_direction1], [hist_texture2, hist_gradient2, hist_direction2], weights)
    print("Matching score:", score)

if __name__ == "__main__":
    main()
