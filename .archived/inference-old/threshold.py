import numpy as np
import cv2


def identify_optimal_thershold(wsi, x1, y1, x2, y2, scale_factor):

    x1 = int(x1 / scale_factor)
    y1 = int(y1 / scale_factor)
    x2 = int(x2 / scale_factor)
    y2 = int(y2 / scale_factor)

    tissue_region = wsi.crop((x1, y1, x2, y2))

    img = tissue_region.convert("L")
    img_array = np.array(img)
    # Adaptive thresholding (uses a small window to determine local thresholds)
    adaptive_thresh = cv2.adaptiveThreshold(
        img_array, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 15
    )
    return int(adaptive_thresh.mean())  # Use the mean as a dynamic threshold