import os
from pathlib import Path
import shutil
from PIL import Image
import cv2
import torch
from collections import defaultdict, Counter
import requests


def calculate_area(box):
    """
    Calculates the area of a bounding box.

    Args:
        box (list or tuple): Bounding box coordinates [x0, y0, x1, y1].

    Returns:
        float: Area of the bounding box.
    """
    x0, y0, x1, y1 = box
    return (x1 - x0) * (y1 - y0)


def filter_boxes(boxes_xyxy, width, height, area_threshold=0.1):
    """
    Filters a list of bounding boxes based on a minimum area threshold and returns the largest box.
    Args:
        boxes_xyxy (list or torch.Tensor): A list or tensor of bounding boxes in (x1, y1, x2, y2) format.
        width (int): The width of the image.
        height (int): The height of the image.
        area_threshold (float, optional): The minimum area threshold as a fraction of the image area. Defaults to 0.1.
    Returns:
        list: A list containing the largest bounding box that meets the area threshold.
              Returns an empty list if no boxes meet the threshold.
    """
    if not boxes_xyxy.size:
        return []

    boxes = boxes_xyxy.tolist() if isinstance(boxes_xyxy, torch.Tensor) else boxes_xyxy
    filtered_boxes = []

    for current_box in boxes:
        current_area = calculate_area(current_box)
        if current_area < width * height * area_threshold:
            continue
        filtered_boxes.append(current_box)

    if not filtered_boxes:
        return []

    largest_box = max(filtered_boxes, key=calculate_area)
    return [largest_box]


def resize_image_if_needed(
    input_img: Image.Image, max_dimension: int = 512
) -> Image.Image:
    w, h = input_img.size
    if h > max_dimension or w > max_dimension:
        if w > h:
            scaling_factor = max_dimension / w
        else:
            scaling_factor = max_dimension / h
        new_width = int(w * scaling_factor)
        new_height = int(h * scaling_factor)
        input_img = input_img.resize((new_width, new_height), Image.LANCZOS)
    return input_img


def save_img_to_dir(img_path, base_dir, img_source=None, format="jpg"):
    """
    Save an image to a directory based on the material type.
    This function creates a directory for the specified material type within the base directory,
    checks if the image file exists and has a valid extension, and then copies the image to the
    appropriate directory with a combined name.
    Args:
        img_path (str): The path to the image file.
        base_dir (str, optional): The base directory where images will be sorted. Defaults to "sorted_images".
        img_source (optional): Image data to save.
        format (str, optional): The image format to save (e.g., "jpg", "png"). Defaults to "jpg".
    Returns:
        None
    """

    supported_formats = ["jpg", "jpeg", "png"]
    format = format.lower()
    if format not in supported_formats:
        print(
            f"Unsupported format: {format}. Supported formats are {supported_formats}"
        )
        return

    # Check if the file exists before copying
    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}")
        return

    parent_dir = Path(img_path).parent.name
    img_name = Path(img_path).stem  # Remove original extension
    new_img_name = f"{img_name}.{format}"

    os.makedirs(os.path.join(base_dir, parent_dir), exist_ok=True)

    # Copy the image to the appropriate directory
    new_img_path = os.path.join(base_dir, parent_dir, new_img_name)
    if img_source is not None and img_source.size > 0:
        if format in ["jpg", "jpeg"] and img_source.shape[2] == 4:
            # Convert BGRA to BGR for JPEG
            img_source = cv2.cvtColor(img_source, cv2.COLOR_BGRA2BGR)
        cv2.imwrite(new_img_path, img_source)
    else:
        shutil.copy(img_path, new_img_path)
    print(f"Copied image to {new_img_path}")


def get_image_count(
    directory, image_extensions=[".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
):
    """
    Retrieves the number of image files in the specified directory.

    Args:
        directory (str): The root directory to search for image files.
        image_extensions (list, optional): A list of image file extensions to look for.
                                           Defaults to ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'].

    Returns:
        int: The number of image files found in the directory.
    """
    image_paths = get_all_images(directory, image_extensions)
    return len(image_paths)


def get_all_images(
    directory, image_extensions=[".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff"]
):
    """
    Recursively retrieves all image file paths from the specified directory.

    Args:
        directory (str): The root directory to search for image files.
        image_extensions (list, optional): A list of image file extensions to look for.
                                           Defaults to ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'].

    Returns:
        list: A list of full file paths to the images found in the directory.

    Raises:
        FileNotFoundError: If the specified directory does not exist.
    """
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory '{directory}' does not exist.")
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_paths.append(os.path.join(root, file))
    return image_paths


def most_common_classification(classification_str):
    classifications = [cls.strip() for cls in classification_str.split(",")]
    if not classifications or classifications == [""]:
        return "none"
    count = Counter(classifications)
    most_common, _ = count.most_common(1)[0]
    return most_common


def split_images_to_directories(input_directory, output_base_directory, num_splits=5):
    # Create output directories
    for i in range(num_splits):
        output_dir = Path(output_base_directory) / f"split_{i+1}"
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get list of image files
    image_files = [
        f
        for f in os.listdir(input_directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".gif"))
    ]

    # Split images into subdirectories
    for idx, image_file in enumerate(image_files):
        src_path = Path(input_directory) / image_file
        dest_dir = Path(output_base_directory) / f"split_{(idx % num_splits) + 1}"
        shutil.copy(src_path, dest_dir / image_file)

    print(f"Images split into {num_splits} directories.")


def copy_image_mask_pairs(
    images_dir, masks_dir, dest_images="data/images", dest_masks="data/mask"
):
    """
    Finds pairs of images and masks based on matching base filenames and
    copies them into new directories.

    Args:
        images_dir (str): Directory path containing images.
        masks_dir (str): Directory path containing masks.
        dest_images (str): Destination directory for the copied images.
        dest_masks (str): Destination directory for the copied masks.
    """
    # Create destination directories if they don't exist.
    os.makedirs(dest_images, exist_ok=True)
    os.makedirs(dest_masks, exist_ok=True)

    # List files in images and masks directories (only common image file extensions).
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".gif", "webp")
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(valid_exts)]
    mask_files = [f for f in os.listdir(masks_dir) if f.lower().endswith(valid_exts)]

    # Create dictionaries mapping base filenames (without extension) to full file names.
    def build_dict(files):
        mapping = {}
        for f in files:
            base, _ = os.path.splitext(f)
            mapping[base] = f
        return mapping

    images_dict = build_dict(image_files)
    masks_dict = build_dict(mask_files)

    # Find common base names.
    common_bases = set(images_dict.keys()) & set(masks_dict.keys())

    print(f"Found {len(common_bases)} matching pairs.")

    # Copy matching image and mask pairs to the destination directories.
    for base in common_bases:
        src_image = Path(images_dir) / images_dict[base]
        src_mask = Path(masks_dir) / masks_dict[base]
        dst_image = Path(dest_images) / images_dict[base]
        dst_mask = Path(dest_masks) / masks_dict[base]

        shutil.copy(src_image, dst_image)
        shutil.copy(src_mask, dst_mask)
        print(f"Copied pair: {images_dict[base]}")


def download_sam_model(model_name: str, dest_dir: str = "models/sam") -> str:
    """
    Downloads the specified SAM model weights to the destination directory.

    Supported model names:
        - sam_vit_h
        - sam_vit_l
        - sam_vit_b

    Args:
        model_name (str): The SAM model name.
        dest_dir (str): Directory to save the downloaded model file.

    Returns:
        str: The full path to the downloaded model file.
    """
    # Mapping model names to download URLs.
    model_urls = {
        "sam_vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sam_vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
    }

    if model_name not in model_urls:
        raise ValueError(
            f"Unsupported model name '{model_name}'. Supported models: {list(model_urls.keys())}"
        )

    os.makedirs(dest_dir, exist_ok=True)
    model_url = model_urls[model_name]
    filename = model_url.split("/")[-1]
    dest_path = Path(dest_dir) / filename

    if dest_path.exists():
        print(f"Model file already exists: {dest_path}")
        return str(dest_path)

    print(f"Downloading {model_name} model from {model_url}...")
    response = requests.get(model_url, stream=True)
    response.raise_for_status()

    with open(dest_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)

    print(f"Downloaded {model_name} to {dest_path}")
    return str(dest_path)


if __name__ == "__main__":
    # directory = "data/classified/clip_images11/other"
    # image_count = get_image_count(directory)
    # print(f"Number of images found: {image_count}")
    # split_images_to_directories("selected_images_1000", "split_images")
    copy_image_mask_pairs(
        "data/images_bash1_copyed",
        "data/mask_folder",
        "data/train1/images",
        "data/train1/masks",
    )
