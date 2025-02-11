import os
import json
import numpy as np
from shapely.geometry import Polygon
from shapely.ops import unary_union
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def base_filename(file_name):
    # Find the last underscore in the file name.
    idx = file_name.rfind("_")
    # Check if ".rf" appears after that underscore.
    if idx != -1 and ".rf" in file_name[idx:]:
        base = file_name[:idx]
    else:
        base = file_name
    return base + ".png"


def save_subtracted_masks(json_path, output_folder):
    """
    Reads annotation data from a JSON file, creates subtracted mask images,
    and saves them with the same filename as the original images in output_folder.

    Args:
        json_path (str): Path to the JSON file with annotation data.
        output_folder (str): Path to the folder where the mask images are saved.
    """
    os.makedirs(output_folder, exist_ok=True)

    # Load JSON data from file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Create a mapping from category_id to category name
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}

    # Loop over each image in the JSON
    for image in data["images"]:
        image_id = image["id"]
        width, height = image["width"], image["height"]
        # Use the same filename as in the JSON for the mask
        file_name = image.get("file_name", f"{image_id}.png")
        # clean_name = base_filename(file_name)

        # Filter annotations for the current image
        image_annotations = [
            ann for ann in data["annotations"] if ann["image_id"] == image_id
        ]

        # Group polygons by category name
        polygons_by_category = {}
        for ann in image_annotations:
            cat_id = ann["category_id"]
            cat_name = cat_id_to_name.get(cat_id, f"cat_{cat_id}")
            for seg in ann["segmentation"]:
                coords = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                poly = Polygon(coords)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                polygons_by_category.setdefault(cat_name, []).append(poly)

        # Merge polygons within each category
        union_polygons = {}
        for cat, poly_list in polygons_by_category.items():
            union_polygons[cat] = unary_union(poly_list)

        # Subtract category "remove" from "rock-0jqu" if both exist
        if "rock" in union_polygons:
            if "remove" in union_polygons:
                try:
                    result_polygon = union_polygons["rock"].difference(
                        union_polygons["remove"]
                    )
                except Exception as e:
                    print(f"Error subtracting for image {image_id}: {e}")
                    continue
            else:
                result_polygon = union_polygons["rock"]
        else:
            print(f"Image {image_id}: Skipping - needed categories not found.")
            continue

        # Create a blank image for the mask
        mask_img = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask_img)

        if not result_polygon.is_empty:
            if result_polygon.geom_type == "Polygon":
                polys_to_draw = [result_polygon]
            elif result_polygon.geom_type == "MultiPolygon":
                polys_to_draw = list(result_polygon.geoms)
            else:
                polys_to_draw = []

            for poly in polys_to_draw:
                exterior_coords = [
                    (int(round(x)), int(round(y))) for x, y in poly.exterior.coords
                ]
                draw.polygon(exterior_coords, outline=255, fill=255)
                for interior in poly.interiors:
                    interior_coords = [
                        (int(round(x)), int(round(y))) for x, y in interior.coords
                    ]
                    draw.polygon(interior_coords, outline=0, fill=0)

        # Save mask to the output folder with the same filename
        output_path = os.path.join(output_folder, file_name)
        mask_img.save(output_path)
        print(f"Saved mask for image_id {image_id} to: {output_path}")


if __name__ == "__main__":
    # Example usage:
    # Adjust these paths to match your setup.
    json_file = "json/_annotations.coco.json"
    destination_folder = "mask_folder"

    save_subtracted_masks(json_file, destination_folder)
