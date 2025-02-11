import cv2
from brisque import BRISQUE


def detect_quality_brisque(image_path):
    image = cv2.imread(image_path)
    brisque = BRISQUE()
    quality_score = brisque.score(image)
    return quality_score


image_path = "your_image.jpg"
quality_score = detect_quality_brisque(image_path)
print(f"BRISQUE Quality Score: {quality_score}")