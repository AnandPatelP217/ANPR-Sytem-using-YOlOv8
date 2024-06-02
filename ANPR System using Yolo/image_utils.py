import cv2

def are_similar(image1, image2, threshold=0.9):
    # Resize images to the same size
    height = min(image1.shape[0], image2.shape[0])
    width = min(image1.shape[1], image2.shape[1])
    image1_resized = cv2.resize(image1, (width, height))
    image2_resized = cv2.resize(image2, (width, height))

    # Convert images to grayscale
    gray1 = cv2.cvtColor(image1_resized, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2_resized, cv2.COLOR_BGR2GRAY)

    # Calculate Structural Similarity Index (SSIM)
    similarity = cv2.matchTemplate(gray1, gray2, cv2.TM_CCOEFF_NORMED)

    return similarity.max() >= threshold
