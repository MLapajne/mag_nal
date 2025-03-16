import os
import numpy as np
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity

# Load a pre-trained ResNet50 model without the top classification layers.
model = ResNet50(weights="imagenet", include_top=False, pooling="avg")


def extract_features(img_path, model, target_size=(224, 224)):
    """Load an image, preprocess it, and extract features using the pre-trained model."""
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = model.predict(img_array)
    return features.flatten()


# Suppose each class (crag) has only one representative image.
# Create your database: mapping from class labels to image paths.
database = {
    "crag1": "path/to/crag1.jpg",
    "crag2": "path/to/crag2.jpg",
    # add more classes
}

# Extract and store features for each class.
features_db = {}
for crag_id, img_path in database.items():
    if os.path.exists(img_path):
        features_db[crag_id] = extract_features(img_path, model)
    else:
        print(f"Image not found: {img_path}")


def find_best_match(query_img_path, model, features_db):
    """Extract features from the query image and match against the database using cosine similarity."""
    query_features = extract_features(query_img_path, model)
    similarities = {}
    for crag_id, feat in features_db.items():
        sim = cosine_similarity([query_features], [feat])[0][0]
        similarities[crag_id] = sim
    best_match = max(similarities, key=similarities.get)
    return best_match, similarities


# Example usage:
query_image_path = "path/to/query.jpg"
if os.path.exists(query_image_path):
    best_crag, similarity_scores = find_best_match(query_image_path, model, features_db)
    print("Best matching crag image:", best_crag)
    print("Similarity scores:", similarity_scores)
else:
    print(f"Query image not found: {query_image_path}")
