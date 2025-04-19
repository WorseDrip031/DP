import rasterio
import geojson
from shapely.geometry import shape, mapping
from rasterio.features import shapes
from pathlib import Path


def convert_masks_to_geojson(masks_folder:Path):

    print("Geojson conversion analysis started...")

    output_file_path = masks_folder / "Annotation.geojson"

    if output_file_path.exists():
        print("Geojson conversion complete...")
        return

    # Paths to the mask files
    mask_paths = {
        "Gleason": masks_folder / "Gleason_Mask.tif",
        "Normal": masks_folder / "Normal_Mask.tif",
        "Stroma": masks_folder / "Stroma_Mask.tif"
    }

    # Initialize list to hold all features
    all_features = []

    # Loop through each mask and create features
    for mask_type, mask_path in mask_paths.items():
        with rasterio.open(mask_path) as src:
            mask = src.read(1)  # Read the first band (single-layer)
            transform = src.transform  # Get affine transformation

        # Convert binary mask to vector polygons
        polygons = []
        for geom, value in shapes(mask, transform=transform):
            if value > 0:  # Extract only foreground (nonzero) regions
                polygons.append(shape(geom))

        # Add features to the list with their corresponding mask type
        for polygon in polygons:
            feature = geojson.Feature(geometry=mapping(polygon), properties={"classification": mask_type})
            all_features.append(feature)

    # Create GeoJSON feature collection
    geojson_data = geojson.FeatureCollection(all_features)

    # Save as GeoJSON file
    with open(output_file_path, "w") as f:
        geojson.dump(geojson_data, f, indent=2)

    print("Geojson conversion complete...")