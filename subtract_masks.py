import json
from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import unary_union
import os
import cv2
import numpy as np
from PIL import Image


def generate_binary_masks_from_coco(
    json_path: str, output_dir: str, mask_suffix: str = "_mask.png", debug: bool = False
) -> None:
    """
    Generates binary mask images from a COCO-like JSON annotation file without requiring the original images.

    :param json_path: Path to the COCO-like JSON file.
    :param output_dir: Directory where mask images will be saved.
    :param mask_suffix: Suffix to append to the original image file name for the mask. Default is "_mask.png".
    :param debug: If True, prints detailed debug information.
    """

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load JSON data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Build a mapping from image_id to image info
    image_info_dict = {image["id"]: image for image in data["images"]}

    # Build a mapping from image_id to its annotations
    annotations_dict = {}
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        annotations_dict.setdefault(image_id, []).append(ann)

    # Iterate over each image in the JSON
    for image_id, image_info in image_info_dict.items():
        file_name = image_info.get("file_name", f"image_{image_id}")
        height = image_info.get("height")
        width = image_info.get("width")

        if debug:
            print(
                f"Processing Image ID: {image_id}, File Name: {file_name}, Size: {width}x{height}"
            )

        # Initialize a blank mask image (single channel, uint8)
        mask = np.zeros((height, width), dtype=np.uint8)

        # Retrieve all annotations for this image
        annotations = annotations_dict.get(image_id, [])

        for ann in annotations:
            segmentation = ann.get("segmentation", [])

            # COCO format: segmentation can be polygons (list of lists)
            if isinstance(segmentation, list):
                for seg in segmentation:
                    if not isinstance(seg, list):
                        continue  # Skip if not a list
                    if len(seg) < 6:
                        # Not enough points to form a polygon
                        if debug:
                            print(
                                f"Skipping invalid segmentation with length {len(seg)}"
                            )
                        continue

                    # Convert the segmentation list to a numpy array of shape (-1, 2)
                    polygon = np.array(seg).reshape(-1, 2)

                    # Ensure the polygon has at least 3 points
                    if polygon.shape[0] < 3:
                        if debug:
                            print(
                                f"Skipping polygon with less than 3 points: {polygon}"
                            )
                        continue

                    # Optionally, validate and fix the polygon using Shapely
                    shapely_poly = Polygon(polygon)
                    if not shapely_poly.is_valid:
                        shapely_poly = shapely_poly.buffer(
                            0
                        )  # Attempt to fix invalid polygons
                        if not shapely_poly.is_valid:
                            if debug:
                                print(f"Invalid polygon could not be fixed: {polygon}")
                            continue
                        polygon = np.array(shapely_poly.exterior.coords).astype(
                            np.int32
                        )
                    else:
                        polygon = polygon.astype(np.int32)

                    # Fill the polygon on the mask with white (255)
                    cv2.fillPoly(mask, [polygon], 255)
            else:
                if debug:
                    print(
                        f"Unsupported segmentation format for annotation ID {ann.get('id')}: {type(segmentation)}"
                    )

        # Define the mask file name
        base_name, _ = os.path.splitext(file_name)
        mask_file_name = f"{base_name}{mask_suffix}"
        mask_path = os.path.join(output_dir, mask_file_name)

        # Save the mask image as PNG
        cv2.imwrite(mask_path, mask)

        if debug:
            print(f"Saved mask to: {mask_path}\n")

    if debug:
        print("All binary masks have been generated successfully.")


def subtract_categories_in_coco(
    input_json_path: str,
    output_json_path: str,
    cat_id_main: int,
    cat_id_sub: int,
    keep_sub_category: bool = False,
    decimal_places: int = 3,  # Number of decimal places to round to
    debug: bool = False,  # Enable debug prints
) -> None:
    """
    Subtract polygons of one category (cat_id_sub) from another category (cat_id_main)
    in a COCO-style JSON annotation file, with proper handling of multiple segmentations.

    :param input_json_path:     Path to the input JSON file.
    :param output_json_path:    Path to save the new JSON file.
    :param cat_id_main:         Category ID from which we subtract.
    :param cat_id_sub:          Category ID that we will subtract out.
    :param keep_sub_category:   If True, keep the original cat_id_sub annotations
                                in the output; if False, exclude them.
    :param decimal_places:      Number of decimal places to round bbox and area values to.
    :param debug:               If True, prints detailed debug information.
    """

    # Helper functions
    def coco_segmentation_to_polygons(segmentation):
        """
        Convert COCO segmentation lists into Shapely Polygons.
        Handles multiple polygons per annotation.

        :param segmentation: List of lists, each containing [x1, y1, x2, y2, ...]
        :return: Shapely Geometry (Polygon or MultiPolygon) or None if invalid
        """
        polygons = []
        for seg in segmentation:
            if not isinstance(seg, list):
                if debug:
                    print(f"Invalid segmentation format: {seg}")
                continue
            if len(seg) < 6:
                # Not enough points to form a polygon
                if debug:
                    print(f"Skipping segmentation with insufficient points: {seg}")
                continue
            points = list(zip(seg[0::2], seg[1::2]))
            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)  # Attempt to fix invalid polygons
                if not poly.is_valid:
                    if debug:
                        print(f"Invalid polygon could not be fixed: {points}")
                    continue
            if not poly.is_empty:
                polygons.append(poly)

        if not polygons:
            return None
        elif len(polygons) == 1:
            return polygons[0]
        else:
            return unary_union(polygons)  # Combine into MultiPolygon

    def polygon_to_coco_segmentation(polygon):
        """
        Convert a Shapely Polygon back to COCO-style segmentation: [[x1,y1,x2,y2,...]]
        """
        if isinstance(polygon, Polygon):
            polygons = [polygon]
        elif isinstance(polygon, MultiPolygon):
            polygons = list(polygon.geoms)
        else:
            return []

        coco_segs = []
        for poly in polygons:
            exterior_coords = list(poly.exterior.coords)
            segmentation = []
            for x, y in exterior_coords:
                segmentation.extend(
                    [round(x, decimal_places), round(y, decimal_places)]
                )
            coco_segs.append(segmentation)
        return coco_segs

    # 1. Load the input JSON
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # We will rebuild the 'annotations' from scratch
    new_annotations = []

    # Keep track of polygons for cat_id_main and cat_id_sub, grouped by image_id
    polygons_by_image_main = {}
    polygons_by_image_sub = {}

    # This will hold any other annotations that we want to keep as-is
    other_annotations = []

    # 2. Split annotations by category
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        seg = ann["segmentation"]
        cat_id = ann["category_id"]

        if cat_id == cat_id_main:
            # Convert to shapely polygon
            poly = coco_segmentation_to_polygons(seg)
            if poly:
                polygons_by_image_main.setdefault(image_id, []).append(poly)
                if debug:
                    print(f"Added main polygon for image {image_id}: {poly}")
        elif cat_id == cat_id_sub:
            # Convert to shapely polygon
            poly = coco_segmentation_to_polygons(seg)
            if poly:
                polygons_by_image_sub.setdefault(image_id, []).append(poly)
                if debug:
                    print(f"Added sub polygon for image {image_id}: {poly}")

                # If we are keeping cat_id_sub in the final output, store these annotations for later
                if keep_sub_category:
                    other_annotations.append(ann)
                    if debug:
                        print(
                            f"Keeping sub-category annotation ID {ann['id']} for image {image_id}"
                        )
        else:
            # Keep all other categories as-is
            other_annotations.append(ann)
            if debug:
                print(
                    f"Keeping other-category annotation ID {ann['id']} for image {image_id}"
                )

    # 3. For each image, subtract sub from main and create new annotations
    # Dynamically determine the starting ID to avoid conflicts
    existing_ids = [ann["id"] for ann in data["annotations"]]
    start_new_id = max(existing_ids) + 1 if existing_ids else 1
    new_id_counter = start_new_id

    # Gather all relevant image_ids from both dictionaries.
    all_image_ids = set(polygons_by_image_main.keys()).union(
        set(polygons_by_image_sub.keys())
    )

    for image_id in all_image_ids:
        main_polygons = polygons_by_image_main.get(image_id, [])
        sub_polygons = polygons_by_image_sub.get(image_id, [])

        if debug:
            print(f"\nProcessing image_id {image_id}:")
            print(f"  Main polygons count: {len(main_polygons)}")
            print(f"  Sub polygons count: {len(sub_polygons)}")

        if not main_polygons:
            # No main polygons to subtract from, so nothing special to do
            if debug:
                print("  No main polygons to subtract from.")
            continue

        # Union the main polygons
        main_union = (
            unary_union(main_polygons) if len(main_polygons) > 1 else main_polygons[0]
        )
        if debug:
            print(f"  Union of main polygons: {main_union}")

        # Union the sub polygons
        if sub_polygons:
            sub_union = (
                unary_union(sub_polygons) if len(sub_polygons) > 1 else sub_polygons[0]
            )
            if debug:
                print(f"  Union of sub polygons: {sub_union}")

            # Subtract sub from main
            new_main_geom = main_union.difference(sub_union)
            if debug:
                print(f"  Resulting geometry after subtraction: {new_main_geom}")
        else:
            # No sub polygons, so new_main_geom is just the main_union
            new_main_geom = main_union
            if debug:
                print(
                    f"  No sub polygons. Resulting geometry is main_union: {new_main_geom}"
                )

        # Could be empty, Polygon, or MultiPolygon
        if new_main_geom.is_empty:
            # Means after subtraction there's nothing left
            if debug:
                print("  Resulting geometry is empty after subtraction.")
            continue

        if new_main_geom.geom_type == "Polygon":
            new_geoms = [new_main_geom]
        elif new_main_geom.geom_type == "MultiPolygon":
            new_geoms = list(new_main_geom.geoms)
        else:
            # Rarely, difference can yield lines or points if edges just touch,
            # which wouldn't be valid in COCO. Usually you can ignore or skip them.
            new_geoms = [g for g in new_main_geom if g.geom_type == "Polygon"]
            if debug:
                print(f"  Skipping non-polygon geometries. Count: {len(new_geoms)}")

        # Convert each resulting polygon back to an annotation
        for poly in new_geoms:
            segmentation = polygon_to_coco_segmentation(poly)
            if not segmentation:
                if debug:
                    print("  Skipping empty or invalid segmentation.")
                continue  # Skip empty or invalid polygons

            x_min, y_min, x_max, y_max = [
                round(val, decimal_places) for val in poly.bounds
            ]
            w = round(x_max - x_min, decimal_places)
            h = round(y_max - y_min, decimal_places)
            area = round(poly.area, decimal_places)

            if debug:
                print(f"  Creating new annotation ID {new_id_counter}:")
                print(f"    BBox: [{x_min}, {y_min}, {w}, {h}]")
                print(f"    Area: {area}")
                print(f"    Segmentation: {segmentation}")

            ann_dict = {
                "id": new_id_counter,
                "image_id": image_id,
                "category_id": cat_id_main,  # same main category
                "segmentation": segmentation,
                "bbox": [x_min, y_min, w, h],
                "area": area,
                "iscrowd": 0,
            }
            new_annotations.append(ann_dict)
            new_id_counter += 1

    # 4. Combine new annotations (for cat_id_main after subtraction) + other
    final_annotations = []
    final_annotations.extend(new_annotations)
    final_annotations.extend(other_annotations)

    if debug:
        print(f"\nTotal new annotations created: {len(new_annotations)}")
        print(f"Total other annotations kept: {len(other_annotations)}")

    # 5. Gather which category IDs are actually used in the new annotations
    used_category_ids = set(ann["category_id"] for ann in final_annotations)

    if debug:
        print(f"Used category IDs: {used_category_ids}")

    # Filter out categories that are never used
    filtered_categories = [
        cat for cat in data["categories"] if cat["id"] in used_category_ids
    ]

    if debug:
        print(f"Filtered categories count: {len(filtered_categories)}")

    # 6. Build the new JSON structure
    new_data = {
        "info": data["info"],
        "licenses": data["licenses"],
        "categories": filtered_categories,
        "images": data["images"],
        "annotations": final_annotations,
    }

    # 7. Save to output JSON
    with open(output_json_path, "w") as f:
        json.dump(new_data, f, indent=2)

    print(f"Subtraction complete. New file saved to: {output_json_path}")


if __name__ == "__main__":
    subtract_categories_in_coco(
        input_json_path="_annotations.coco.json",
        output_json_path="_annotations.coco_new.json",
        cat_id_main=1,
        cat_id_sub=2,
        keep_sub_category=False,
    )

    generate_binary_masks_from_coco(
        json_path="_annotations.coco_new.json",
        output_dir="masks",
        mask_suffix="_binary_mask.png",  # Optional: customize mask file suffix
        debug=True,  # Set to True to enable debug prints
    )
