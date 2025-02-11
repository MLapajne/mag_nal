import json
from shapely.geometry import Polygon
from shapely.ops import unary_union


def polygon_to_coco_segmentation(poly):
    """
    Convert a shapely polygon into a COCO segmentation.
    This function uses the exterior coordinates only and returns
    a list (COCO format requires a list of lists).
    """
    # COCO segmentation expects a flattened list of coordinates.
    coords = []
    for x, y in poly.exterior.coords:
        # Rounding coordinates to integers (you can also keep them as floats)
        coords.extend([int(round(x)), int(round(y))])
    return [coords]


def polygon_to_bbox(poly):
    """
    Compute the bounding box [x, y, width, height] from a shapely polygon.
    """
    minx, miny, maxx, maxy = poly.bounds
    x = int(round(minx))
    y = int(round(miny))
    w = int(round(maxx - minx))
    h = int(round(maxy - miny))
    return [x, y, w, h]


def process_images_to_coco(input_json_path, output_json_path):
    # Load input JSON (your original file with segmentation data)
    with open(input_json_path, "r") as f:
        data = json.load(f)

    # Build a new COCO dictionary for the output.
    # Here we define our own info, licenses, and a single category called "rock"
    coco_output = {
        "info": {
            "year": "2025",
            "version": "1",
            "description": "Difference polygons from subtracting 'niki2' from 'niki'",
            "contributor": "",
            "url": "",
            "date_created": "2025-02-03T21:04:47+00:00",
        },
        "licenses": [
            {
                "id": 1,
                "url": "https://creativecommons.org/licenses/by/4.0/",
                "name": "CC BY 4.0",
            }
        ],
        "categories": [
            # In this example, we output a single category with id 1 and name "rock"
            {"id": 1, "name": "rock", "supercategory": "rock"}
        ],
        "images": [],
        "annotations": [],
    }

    # We'll assign annotation IDs sequentially.
    annotation_id = 0

    # Create a mapping from the input category_id to its name.
    # (We assume the input JSON has a "categories" list.)
    cat_id_to_name = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    # Process each image in the input JSON.
    for image in data["images"]:
        image_id = image["id"]
        width = image["width"]
        height = image["height"]

        # Add the image entry as-is to the output.
        coco_output["images"].append(image)

        # Filter annotations for this image.
        image_annotations = [
            ann for ann in data["annotations"] if ann["image_id"] == image_id
        ]

        # Group polygons by category name.
        # We need the polygons for category "niki" and "niki2" in order to perform subtraction.
        polygons_by_category = {}
        for ann in image_annotations:
            cat_id = ann["category_id"]
            cat_name = cat_id_to_name.get(cat_id, f"cat_{cat_id}")
            # Each annotation's segmentation is assumed to be a list of polygon(s)
            for seg in ann["segmentation"]:
                # Convert the flat list [x1, y1, x2, y2, ...] into a list of (x, y) tuples.
                coords = [(seg[i], seg[i + 1]) for i in range(0, len(seg), 2)]
                poly = Polygon(coords)
                if not poly.is_valid:
                    # Fix invalid polygons (if needed)
                    poly = poly.buffer(0)
                polygons_by_category.setdefault(cat_name, []).append(poly)

        # For each category, join the polygons using unary_union.
        # If the polygons are separated, the union will be a MultiPolygon.
        union_polygons = {}
        for cat, poly_list in polygons_by_category.items():
            union_polygons[cat] = unary_union(poly_list)

        # Only if both "niki" and "niki2" are present do we proceed.
        if "rock-0jqu" in polygons_by_category and "remove" in polygons_by_category:
            try:
                # Subtract the union for niki2 from the union for niki.
                # The result may be a Polygon or a MultiPolygon.
                result_polygon = union_polygons["rock-0jqu"].difference(
                    union_polygons["remove"]
                )
            except Exception as e:
                print(f"Error subtracting polygons for image {image_id}: {e}")
                continue
        else:
            print(f"Skipping image {image_id}: missing 'niki' or 'niki2'")
            continue

        if result_polygon.is_empty:
            print(f"Image {image_id}: subtraction result is empty")
            continue

        # The result might be a single Polygon or a MultiPolygon.
        if result_polygon.geom_type == "Polygon":
            polygons_to_write = [result_polygon]
        elif result_polygon.geom_type == "MultiPolygon":
            polygons_to_write = list(result_polygon.geoms)
        else:
            polygons_to_write = []

        # For each polygon in the result, compute segmentation, bbox, and area, then add an annotation.
        for poly in polygons_to_write:
            segmentation = polygon_to_coco_segmentation(poly)
            bbox = polygon_to_bbox(poly)
            area = int(round(poly.area))

            annotation = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,  # All resulting polygons are in category "rock"
                "bbox": bbox,
                "area": area,
                "segmentation": segmentation,
                "iscrowd": 0,
            }
            coco_output["annotations"].append(annotation)
            annotation_id += 1

    # Write the output COCO JSON to file.
    with open(output_json_path, "w") as f:
        json.dump(coco_output, f, indent=2)
    print(f"Output written to {output_json_path}")


if __name__ == "__main__":
    input_json = "_annotations.coco2.json"  # Change this to your input JSON path
    output_json = "difference_output.json"  # Output JSON file
    process_images_to_coco(input_json, output_json)
