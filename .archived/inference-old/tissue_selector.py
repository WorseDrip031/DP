import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from PIL import Image


def onselect(eclick, erelease):
    """Callback function to capture the selected rectangular area."""
    global x1, y1, x2, y2
    x1, y1 = int(eclick.xdata), int(eclick.ydata)
    x2, y2 = int(erelease.xdata), int(erelease.ydata)
    print(f"Selected region: ({x1}, {y1}) to ({x2}, {y2})")


def select_region(image_path):
    """Displays an image and allows the user to select a rectangular region."""
    global x1, y1, x2, y2
    x1 = y1 = x2 = y2 = 0  # Initialize selection coordinates
    
    image = Image.open(image_path)
    image_array = np.array(image)
    
    _ , ax = plt.subplots()
    ax.imshow(image_array)
    ax.set_title("Select a region and close the window when done")
    
    # Enable interactive rectangle selection (without drawtype)
    _ = RectangleSelector(ax, onselect, useblit=True, button=[1], interactive=True)
    
    plt.show()
    
    # Crop the selected region from the image
    if x1 != x2 and y1 != y2:
        
        if image is not None:
            image.close()

        return x1, y1, x2, y2
        # cropped = image.crop((x1, y1, x2, y2))
        # cropped.show()  # Display the selected region
        # return cropped
    else:
        print("No region selected.")
        return None