import os
import random
import shutil


def select_and_copy_images(src_dir, dest_dir, num_images=1000):
    # Ensure the source directory exists
    if not os.path.exists(src_dir):
        raise ValueError(f"Source directory {src_dir} does not exist.")

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Get all image file names in the source directory
    all_images = [
        f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))
    ]
    if len(all_images) < num_images:
        raise ValueError(
            f"Not enough images in the source directory. Found {len(all_images)} images."
        )

    # Randomly select the desired number of images
    selected_images = random.sample(all_images, num_images)

    # Copy the selected images to the destination directory
    for image in selected_images:
        shutil.copy(os.path.join(src_dir, image), os.path.join(dest_dir, image))

    print(f"Successfully copied {num_images} images to {dest_dir}.")


if __name__ == "__main__":
    src_directory = "clip_filtered_images_9/outdoor crag/downloaded_images"
    dest_directory = "selected_images_1000"
    select_and_copy_images(src_directory, dest_directory, num_images=1000)
