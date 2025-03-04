import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from tqdm import tqdm


def create_plots(input_folder_path, plots_folder_path):
    files = sorted(list(input_folder_path.rglob("*.png")))

    # Number of columns
    cols = 5
    rows = -(-len(files) // cols)

    # Set figure size
    _, axes = plt.subplots(rows, cols, figsize=(15, rows * 3))

    # Flatten axes for easy iteration (if only 1 row, axes might not be an array)
    axes = axes.flatten() if rows > 1 else [axes]

    # Loop through files and plot them
    for i, file in enumerate(files):
        img = Image.open(file)
        
        # Extract the relevant numbers from the filename
        parts = file.stem.split('_')
        scale = parts[0]
        sigma = parts[1]
        min_size = parts[2]
        tissue_percentage = parts[3]
        
        # Set the title based on the extracted values
        title = f"Scale:{scale} Sigma:{sigma} Min-Size:{min_size} Tissue:{tissue_percentage}"
        
        axes[i].imshow(img, cmap="gray")  # Assuming black & white images
        axes[i].axis("off")  # Hide axis
        axes[i].set_title(title, fontsize=8)

    # Hide unused subplots (if images are not a multiple of 5)
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    # Save the plot to a file (e.g., PNG format)
    output_path = plots_folder_path / f"{input_folder_path.name}.png"
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")  # Save with tight bounding box to avoid cutting off titles


##### Main #####
folder_path = Path("superpixel_experiments/output")
plots_folder_path = folder_path / "plots"
plots_folder_path.mkdir(parents=True, exist_ok=True)
for sub_folder in tqdm(folder_path.iterdir(), desc="Processing subfolders", unit="folder"):
    if sub_folder.is_dir():  # Optional: only process directories
        create_plots(sub_folder, plots_folder_path)