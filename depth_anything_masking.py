from transformers import pipeline, DepthAnythingConfig
from PIL import Image
import numpy as np
import cv2
import os
from PIL import Image


class DepthAnythingProcessor:
    def __init__(self, image_dir):
        self.image_dir = image_dir

    def process_images(self):

        for filename in os.listdir(self.image_dir):
            if filename.endswith((".png", ".jpg", ".jpeg", ".webp")):
                image_path = os.path.join(self.image_dir, filename)
                image = Image.open(image_path)
                depth_image = self.create_depth_image(image)
                mask = self.create_mask(depth_image)
                masked_image = self.apply_mask(image, mask)
                masked_image.save(os.path.join(self.image_dir, f"masked_{filename}"))

    def create_depth_image(self, image):
        config = DepthAnythingConfig(
            depth_estimation_type="metric",
            max_depth=80,  # Set this to the appropriate value for your use case
        )
        pipe = pipeline(
            task="depth-estimation",
            model="depth-anything/Depth-Anything-V2-Small-hf",
            config=config,
        )
        depth_image = pipe(image)["depth"]
        return depth_image

    def create_mask(self, depth_image):

        depth_array = np.array(depth_image)

        # Apply custom thresholding
        custom_threshold = depth_array.mean() * 0.3  # Adjust this factor as needed
        _, mask = cv2.threshold(depth_array, custom_threshold, 255, cv2.THRESH_BINARY)

        # Convert mask to boolean array
        mask = mask.astype(bool)

        return mask

    def apply_mask(self, image, mask):
        image_array = np.array(image)
        masked_array = image_array.copy()
        masked_array[~mask] = 0  # Set background pixels to black
        masked_image = Image.fromarray(masked_array)
        return masked_image


# Example usage
if __name__ == "__main__":
    image_directory = "par_slik"
    processor = DepthAnythingProcessor(image_directory)
    processor.process_images()
    """
    img_proccesed = processor.create_depth_image(
        Image.open(
            "par_slik/Ditter-Country:Germany-Ascents:0-Karma:425_a280beb5cd1d1a1548f617820185762e0fac6ac4.jpg"
        )
    )
    img_proccesed.show()
    """
