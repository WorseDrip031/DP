import cv2
import numpy as np
from pathlib import Path


def finetune_segmentation(wsi_path:Path,
                          mask_path:Path,
                          output_path:Path):
    
    image = cv2.imread(wsi_path, cv2.IMREAD_COLOR)

    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

    _, PFG = cv2.threshold(mask_image, 245, 255, cv2.THRESH_BINARY)

    kernel = np.ones((16, 16), np.uint8)
    sure_fg = cv2.erode(mask_image, kernel, iterations=4) 
    _, FG = cv2.threshold(sure_fg, 245, 255, cv2.THRESH_BINARY)

    sure_bg = cv2.dilate(mask_image, kernel, iterations=4)
    _, BG = cv2.threshold(sure_bg, 245, 255, cv2.THRESH_BINARY)
    BG = cv2.bitwise_not(BG)

    PBG = np.ones(image.shape[:2], np.uint8)
    PBG[PFG == 255] = 0
    PBG[FG == 255] = 0
    PBG[BG == 255] = 0

    mask[:, :] = 3
    mask[BG == 255] = 0
    mask[FG == 255] = 1
    mask[PBG == 255] = 2

    print("GC started...")
    mask, bgdModel, fgdModel = cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    binary_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')

    cv2.imwrite(output_path, binary_mask)
