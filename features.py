
import cv2
import numpy as np
IMAGE_SIZE = (256, 256)
def extract_features(image):
    """
    Extract color and texture features from a BGR leaf image.

    Features:
    - HSV color histograms
    - LAB color histograms
    - Grayscale histogram
    - Texture statistics
    Returns a one-dimensional NumPy feature vector.
    """
    if image is None or image.size == 0:
        raise ValueError("Input image is empty or invalid.")
    # 1. Resize image consistently
    image = cv2.resize(
        image,
        IMAGE_SIZE,
        interpolation=cv2.INTER_AREA
    )
    # 2. Create a basic leaf-region mask
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    green_mask = (
        (hue >= 25) &
        (hue <= 95) &
        (saturation > 25)
    )
    yellow_brown_mask = (
        (hue >= 5) &
        (hue < 35) &
        (saturation > 35) &
        (value > 25)
    )
    mask = (
        green_mask | yellow_brown_mask
    ).astype(np.uint8) * 255
    # Clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )
    # Use the largest connected region when available
    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask
    )
    if num > 1:
        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )
        mask = (
            labels == largest
        ).astype(np.uint8) * 255
    # If mask is too small, fall back to the whole image
    if np.count_nonzero(mask) < 100:
        mask = None
    features = []
    # 3. HSV histograms
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_settings = [
        (0, 16, [0, 180]),
        (1, 16, [0, 256]),
        (2, 16, [0, 256])
    ]
    for channel, bins, limits in hsv_settings:
        histogram = cv2.calcHist(
            [hsv],
            [channel],
            mask,
            [bins],
            limits
        )
        histogram = cv2.normalize(
            histogram,
            histogram
        ).flatten()
        features.extend(histogram)
    # 4. LAB color histograms
    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )
    for channel in range(3):
        histogram = cv2.calcHist(
            [lab],
            [channel],
            mask,
            [16],
            [0, 256]
        )
        histogram = cv2.normalize(
            histogram,
            histogram
        ).flatten()
        features.extend(histogram)
    # 5. Grayscale texture features
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )
    gray_hist = cv2.calcHist(
        [gray],
        [0],
        mask,
        [16],
        [0, 256]
    )
    gray_hist = cv2.normalize(
        gray_hist,
        gray_hist
    ).flatten()
    features.extend(gray_hist)
    # 6. Texture statistics
    if mask is not None:
        gray_pixels = gray[mask > 0]
    else:
        gray_pixels = gray.flatten()
    features.extend([
        np.mean(gray_pixels),
        np.std(gray_pixels),
        np.var(gray_pixels),
        cv2.Laplacian(
            gray,
            cv2.CV_64F
        ).var()
    ])
    return np.asarray(
        features,
        dtype=np.float32
    )
