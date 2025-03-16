import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, Input
from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K


# -------------------------------
# 1. Define the Base Network
# -------------------------------
def create_base_network(input_shape):
    """
    A simple CNN to extract features from an image.
    """
    input = Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation="relu")(input)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    return models.Model(input, x)


# -------------------------------
# 2. Define a Lambda Layer for Distance
# -------------------------------
def euclidean_distance(vects):
    """
    Compute the Euclidean distance between two vectors.
    """
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


# -------------------------------
# 3. Define the Contrastive Loss Function
# -------------------------------
def contrastive_loss(y_true, y_pred):
    """
    Contrastive loss from Hadsell-et-al.'06:
    L = (1 - y_true) * 0.5 * (D)^2 + y_true * 0.5 * {max(margin - D, 0)}^2
    Here, y_true=0 for similar pairs and 1 for dissimilar pairs.
    """
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean((1 - y_true) * square_pred + y_true * margin_square)


# -------------------------------
# 4. Build the Siamese Network Model
# -------------------------------
input_shape = (28, 28, 1)  # For MNIST; adjust as needed for your images
base_network = create_base_network(input_shape)

# Define the two inputs
input_a = Input(shape=input_shape)
input_b = Input(shape=input_shape)

# Pass both inputs through the same network (shared weights)
processed_a = base_network(input_a)
processed_b = base_network(input_b)

# Compute the Euclidean distance between the two feature vectors
distance = layers.Lambda(euclidean_distance)([processed_a, processed_b])

# Create the complete Siamese network model
siamese_model = models.Model([input_a, input_b], distance)

# Compile the model with the contrastive loss
siamese_model.compile(loss=contrastive_loss, optimizer="adam")

# -------------------------------
# 5. Prepare the Data (Using MNIST as an Example)
# -------------------------------
# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess data: normalize and add channel dimension
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)


def create_pairs(x, digit_indices):
    """
    Positive pairs: two images of the same class.
    Negative pairs: two images from different classes.
    """
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(10)]) - 1
    for d in range(10):
        for i in range(n):
            # positive pair (same class)
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            labels += [0]  # 0 for similar

            # negative pair (different classes)
            inc = (d + np.random.randint(1, 10)) % 10
            z1 = digit_indices[d][i]
            z2 = digit_indices[inc][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1]  # 1 for dissimilar
    return np.array(pairs), np.array(labels)


# Create pairs for training and testing
digit_indices = [np.where(y_train == i)[0] for i in range(10)]
tr_pairs, tr_y = create_pairs(x_train, digit_indices)

digit_indices_test = [np.where(y_test == i)[0] for i in range(10)]
te_pairs, te_y = create_pairs(x_test, digit_indices_test)

# -------------------------------
# 6. Train the Siamese Network
# -------------------------------
siamese_model.fit(
    [tr_pairs[:, 0], tr_pairs[:, 1]],
    tr_y,
    batch_size=128,
    epochs=10,
    validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
)


# -------------------------------
# 7. Evaluate the Model
# -------------------------------
def compute_accuracy(y_true, y_pred, threshold=0.5):
    """
    Compute classification accuracy with a fixed threshold on distances.
    """
    pred = y_pred.ravel() < threshold
    return np.mean(pred == y_true)


# Predict distances on the test pairs
pred = siamese_model.predict([te_pairs[:, 0], te_pairs[:, 1]])
test_accuracy = compute_accuracy(te_y, pred)
print("Test accuracy: {:.2f}%".format(test_accuracy * 100))
