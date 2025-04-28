import cv2
import numpy as np
from pathlib import Path


mask_path = Path("experimentation") / "test" / "Stroma_Mask_cropped.tif"
original_image_path = Path("experimentation") / "test" / "Subset1_Test_001_cropped.tiff"


image = cv2.imread(original_image_path, cv2.IMREAD_COLOR)
#cv2.imshow('Original', image)

mask = np.zeros(image.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

mask_image = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
#cv2.imshow('Mask', mask_image)

_, PFG = cv2.threshold(mask_image, 245, 255, cv2.THRESH_BINARY)
#cv2.imshow('PFG', PFG)

kernel = np.ones((128, 128), np.uint8)
sure_fg = cv2.erode(mask_image, kernel, iterations=4) 
_, FG = cv2.threshold(sure_fg, 245, 255, cv2.THRESH_BINARY)
#cv2.imshow('FG', FG)

sure_bg = cv2.dilate(mask_image, kernel, iterations=4)
_, BG = cv2.threshold(sure_bg, 245, 255, cv2.THRESH_BINARY)
BG = cv2.bitwise_not(BG)
#cv2.imshow('BG', BG)

PBG = np.ones(image.shape[:2], np.uint8)
PBG[PFG == 255] = 0
PBG[FG == 255] = 0
PBG[BG == 255] = 0
#cv2.imshow('PBG', PBG)

mask[:, :] = 3
mask[BG == 255] = 0
mask[FG == 255] = 1
mask[PBG == 255] = 2

mask, bgdModel, fgdModel = cv2.grabCut(image, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
binary_mask = np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')
mask = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
img = image * mask[:, :, np.newaxis]

#cv2.imshow("Segmented", img)
#cv2.imshow('Mask', binary_mask)

output_mask_path = Path("experimentation") / "test" / "Stroma_Mask_cropped_finetuned.tif"

cv2.imwrite(output_mask_path, binary_mask)
