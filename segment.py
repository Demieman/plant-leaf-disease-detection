
"""
K-Means leaf color segmentation and potential discoloration estimation.

GrabCut: separates the leaf from the background using green pixels
as initial foreground markers.

K-Means: groups the colors within the detected leaf.

HSV analysis: estimates potentially yellow and brown regions.

Important:
This is not a plant disease diagnosis or validated severity measurement.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


# -------------------------------------------------
# CONFIGURATION
# -------------------------------------------------

MAX_IMAGE_SIZE = 800
MIN_LEAF_PIXELS = 100
MIN_REGION_AREA = 6

YELLOW_HUE_MIN = 15
YELLOW_HUE_MAX = 35

BROWN_HUE_MIN = 5
BROWN_HUE_MAX = 22


def analyze(image_bgr, clusters=3):

    # 1. Validate input
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Invalid image.")

    if len(image_bgr.shape) != 3 or image_bgr.shape[2] != 3:
        raise ValueError("Expected a three-channel color image.")

    if not 2 <= clusters <= 10:
        raise ValueError("Clusters must be between 2 and 10.")

    # 2. Resize large images
    h, w = image_bgr.shape[:2]

    scale = min(1.0, MAX_IMAGE_SIZE / max(w, h))

    if scale < 1:
        image_bgr = cv2.resize(
            image_bgr,
            (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA
        )

    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # -------------------------------------------------
    # 3. IDENTIFY GREEN LEAF SEEDS
    # -------------------------------------------------

    green_seeds = (
        (hue >= 25) &
        (hue <= 95) &
        (saturation > 30) &
        (value > 25)
    )

    seed_mask = green_seeds.astype(np.uint8) * 255

    seed_mask = cv2.morphologyEx(
        seed_mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (7, 7)
        )
    )

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        seed_mask,
        connectivity=8
    )

    if num <= 1:
        raise ValueError(
            "Could not identify a green leaf region. "
            "Try an image with a clearly visible green leaf."
        )

    largest = 1 + np.argmax(
        stats[1:, cv2.CC_STAT_AREA]
    )

    seed_mask = (
        (labels == largest).astype(np.uint8) * 255
    )

    if np.count_nonzero(seed_mask) < MIN_LEAF_PIXELS:
        raise ValueError("The detected leaf region is too small.")

    # -------------------------------------------------
    # 4. LEAF SEGMENTATION USING GRABCUT
    # -------------------------------------------------

    ys, xs = np.where(seed_mask > 0)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()

    # Expand the region to include potentially damaged edges
    margin_x = max(10, int((x2 - x1 + 1) * 0.20))
    margin_y = max(10, int((y2 - y1 + 1) * 0.20))

    x1 = max(0, x1 - margin_x)
    y1 = max(0, y1 - margin_y)

    x2 = min(image_bgr.shape[1] - 1, x2 + margin_x)
    y2 = min(image_bgr.shape[0] - 1, y2 + margin_y)

    grabcut_mask = np.full(
        image_bgr.shape[:2],
        cv2.GC_BGD,
        dtype=np.uint8
    )

    # Probable foreground inside expanded region
    grabcut_mask[y1:y2 + 1, x1:x2 + 1] = cv2.GC_PR_FGD

    # Definite foreground: green seed pixels
    grabcut_mask[seed_mask > 0] = cv2.GC_FGD

    try:
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)

        cv2.grabCut(
            image_bgr,
            grabcut_mask,
            None,
            bg_model,
            fg_model,
            5,
            cv2.GC_INIT_WITH_MASK
        )

        leaf = np.where(
            (grabcut_mask == cv2.GC_FGD) |
            (grabcut_mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype(np.uint8)

    except cv2.error:
        print("GrabCut failed. Using green seed fallback.")
        leaf = seed_mask.copy()

    # -------------------------------------------------
    # 5. CLEAN LEAF MASK
    # -------------------------------------------------

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        leaf,
        connectivity=8
    )

    if num > 1:
        largest = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        leaf = (
            (labels == largest).astype(np.uint8) * 255
        )

    leaf = cv2.morphologyEx(
        leaf,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )
    )

    leaf_pixels = leaf > 0

    leaf_pixel_count = int(
        np.count_nonzero(leaf_pixels)
    )

    if leaf_pixel_count < MIN_LEAF_PIXELS:
        raise ValueError(
            "Could not isolate a sufficient leaf region."
        )

    # -------------------------------------------------
    # 6. K-MEANS COLOR SEGMENTATION
    # -------------------------------------------------

    lab = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2LAB
    )

    pixels = lab[leaf_pixels].reshape(-1, 3)

    n = min(clusters, len(pixels))

    kmeans = KMeans(
        n_clusters=n,
        random_state=42,
        n_init=10
    )

    group = kmeans.fit_predict(pixels)

    centers_lab = np.uint8(
        np.clip(kmeans.cluster_centers_, 0, 255)
    ).reshape(1, n, 3)

    centers_rgb = cv2.cvtColor(
        centers_lab,
        cv2.COLOR_LAB2RGB
    ).reshape(n, 3)

    segmented = np.full_like(rgb, 245)

    segmented[leaf_pixels] = centers_rgb[group]

    # -------------------------------------------------
    # 7. POTENTIAL DISCOLORATION DETECTION
    # -------------------------------------------------

    yellow_pixels = (
        (hue >= YELLOW_HUE_MIN) &
        (hue <= YELLOW_HUE_MAX) &
        (saturation >= 50) &
        (value >= 60)
    )

    brown_pixels = (
        (hue >= BROWN_HUE_MIN) &
        (hue < BROWN_HUE_MAX) &
        (saturation >= 50) &
        (value >= 30) &
        (value <= 190)
    )

    potential_discoloration = (
        yellow_pixels | brown_pixels
    )

    # Restrict candidates to detected leaf pixels
    potential_discoloration &= leaf_pixels

    # -------------------------------------------------
    # 8. REDUCE EDGE FALSE POSITIVES
    # -------------------------------------------------

    interior_mask = cv2.erode(
        leaf,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (3, 3)
        ),
        iterations=1
    )

    interior = interior_mask > 0

    potential_discoloration &= interior

    abnormal = (
        potential_discoloration.astype(np.uint8) * 255
    )

    # Remove tiny isolated detections
    abnormal = cv2.morphologyEx(
        abnormal,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8)
    )

    # -------------------------------------------------
    # 9. REMOVE VERY SMALL REGIONS
    # -------------------------------------------------

    num, labels, stats, _ = cv2.connectedComponentsWithStats(
        abnormal,
        connectivity=8
    )

    cleaned = np.zeros_like(abnormal)

    for i in range(1, num):

        area = stats[i, cv2.CC_STAT_AREA]

        if area >= MIN_REGION_AREA:
            cleaned[labels == i] = 255

    abnormal = cleaned

    abnormal[~leaf_pixels] = 0

    # -------------------------------------------------
    # 10. GENERATE HIGHLIGHTED OVERLAY
    # -------------------------------------------------

    overlay = rgb.copy()

    abnormal_mask = abnormal > 0

    red_color = np.array(
        [255, 45, 55],
        dtype=np.float32
    )

    overlay[abnormal_mask] = (
        0.45 * overlay[abnormal_mask].astype(np.float32)
        + 0.55 * red_color
    ).astype(np.uint8)

    contours, _ = cv2.findContours(
        abnormal,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    cv2.drawContours(
        overlay,
        contours,
        -1,
        (255, 40, 40),
        2
    )

    # -------------------------------------------------
    # 11. CALCULATE PIXEL STATISTICS
    # -------------------------------------------------

    affected_pixel_count = int(
        np.count_nonzero(abnormal)
    )

    affected_percentage = (
        affected_pixel_count / leaf_pixel_count
    ) * 100

    yellow_count = int(
        np.count_nonzero(yellow_pixels & leaf_pixels)
    )

    brown_count = int(
        np.count_nonzero(brown_pixels & leaf_pixels)
    )

    print("\n========== SEGMENTATION REPORT ==========")

    print("Total leaf pixels:", leaf_pixel_count)
    print("Yellow candidates:", yellow_count)
    print("Brown candidates:", brown_count)
    print("Final flagged pixels:", affected_pixel_count)

    print(
        "Potential discoloration:",
        round(affected_percentage, 2),
        "%"
    )

    print("K-Means clusters:", n)
    print("=========================================\n")

    # -------------------------------------------------
    # 12. RETURN RESULTS
    # -------------------------------------------------

    return {
        "original": rgb,
        "segmented": segmented,
        "overlay": overlay,
        "leaf_mask": leaf,
        "abnormal_mask": abnormal,
        "leaf_pixels": leaf_pixel_count,
        "affected_pixels": affected_pixel_count,
        "affected_percent": round(affected_percentage, 2),
        "clusters": n
    }
