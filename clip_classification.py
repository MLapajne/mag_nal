import os
import torch
import clip
from PIL import Image
from helper import get_all_images, save_img_to_dir


class ImageClassifier:
    def __init__(self, device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model, self.preprocess = clip.load("ViT-L/14", device=self.device)

    def classify_and_copy_images(
        self, input_directory: str, output_directory: str, text_prompts: list
    ):
        """
        Classifies images in the input directory based on text prompts and copies them to respective directories.
        Args:
            input_directory (str): The directory containing the images to be classified.
            output_directory (str): The directory where classified images will be copied to.
            text_prompts (list of str): A list of text prompts used for classifying the images.
        """
        os.makedirs(output_directory, exist_ok=True)
        for prompt in text_prompts:
            os.makedirs(os.path.join(output_directory, prompt), exist_ok=True)

        # Get all image paths
        images = get_all_images(input_directory)
        if not images:
            print("No images found in the input directory.")
            return

        for image_path in images:
            # Classify the image
            label = self.classify_image(image_path, text_prompts)
            dest_path = os.path.join(output_directory, label)
            save_img_to_dir(image_path, dest_path)

        print("Images have been classified and copied to respective directories.")

    def classify_image(
        self, image_path: str, text_prompts: list, threshold: float = 0.18
    ) -> str:
        """
        Classifies an image based on given text prompts using the CLIP model.
        Args:
            image_path (str): The file path to the image to be classified.
            text_prompts (list of str): A list of text prompts to classify the image against.
            threshold (float): The minimum similarity score to accept a classification.
        Returns:
            str: The text prompt that best matches the image.
        """
        image = self.preprocess(Image.open(image_path)).unsqueeze(0).to(self.device)
        text_inputs = clip.tokenize(text_prompts).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarity_scores = (image_features @ text_features.T).squeeze()

        max_score, max_idx = similarity_scores.max(dim=0)
        predicted_class = text_prompts[max_idx.item()]
        print(f"Max score: {max_score.item()}, Max index: {max_idx.item()}")
        if max_score.item() < threshold:
            return "other"
        return predicted_class

    def classify_and_copy_images(
        self, input_directory: str, output_directory: str, text_prompts: list
    ):
        """
        Classifies images in the input directory based on text prompts and copies them to respective directories.
        Args:
            input_directory (str): The directory containing the images to be classified.
            output_directory (str): The directory where classified images will be copied to.
            text_prompts (list of str): A list of text prompts used for classifying the images.
        """
        os.makedirs(output_directory, exist_ok=True)
        for prompt in text_prompts:
            os.makedirs(os.path.join(output_directory, prompt), exist_ok=True)

        # Get all image paths
        images = get_all_images(input_directory)
        if not images:
            print("No images found in the input directory.")
            return

        for image_path in images:
            # Classify the image
            label = self.classify_image(image_path, text_prompts)
            dest_path = os.path.join(output_directory, label)
            save_img_to_dir(image_path, dest_path)

        print("Images have been classified and copied to respective directories.")


# Usage example:
if __name__ == "__main__":
    input_directory = r"data/classified/groundingdino_buildings_4/detections_cropped"
    output_dir_selected = r"data/classified/building_windows_doors_facade"

    # Create an instance of the classifier
    classifier = ImageClassifier(device="cuda")

    # Define text prompts for classification
    text_prompts = [
        "outdoor crag",
        "man-made wall, plastic grips",
        "other",
    ]

    classifier.classify_and_copy_images(
        r"downloaded_images",
        r"clip_filtered_images_9",
        text_prompts,
    )
