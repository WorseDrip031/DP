import numpy as np
from skimage.transform import resize
from skimage.io import imsave, imread

def upsample_wsi(original_wsi, downsampled_wsi, upsampled_wsi_path):

    print("Upsampling analysis started...")

    if upsampled_wsi_path.exists():
        upsampled_wsi = imread(upsampled_wsi_path).astype(np.uint8)
        print("Upsampling complete...")
        return upsampled_wsi

    width, height = original_wsi.size
    new_size = (height, width)

    print("Upsampling started...")
    upsampled_wsi = resize(downsampled_wsi, new_size, preserve_range=True).astype(np.uint8)

    print("Upsampled wsi saving...")
    imsave(upsampled_wsi_path, upsampled_wsi.astype(np.uint8))

    return upsampled_wsi