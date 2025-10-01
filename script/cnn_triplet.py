import os
import cv2
import time
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.optimizers import Adam
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import json
import random

# --- Main Configuration ---
# (Using one dataset for this example)
dataset_name = "sorted_dataset_scale_exclude_45"
output_dir = f"../data/triplet_loss_output/{dataset_name}"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

confidence_rate = 0.4

config = {
    "dataset_root": f"../data/{dataset_name}/dataset",
    "test_data_dir": f"../data/{dataset_name}/test_dataset",
    "confidence_map_dir": "../data/cut_confidence_image",
    "img_size": (48, 48),
    "batch_size": 8,
    "num_classes": 10,
    "epochs": 50, # Increased epochs for metric learning
    "embedding_dim": 128, # Dimension of the output vector
    "triplet_alpha": 0.2, # Margin for the triplet loss
    "learning_rate": 0.0001,
    "model_name": "embedding_model.h5",
    "config_file_name": "config.json",
    "confidence_rate": confidence_rate
}

with open(os.path.join(output_dir, config["config_file_name"]), "w") as file:
    json.dump(config, file, indent=4)

# --- Custom Confidence Dropout Layer (Unchanged) ---
class ConfidenceDropout(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(ConfidenceDropout, self).__init__(**kwargs)

    def call(self, inputs):
        image, confidence_map = inputs
        mask = tf.where(confidence_map >= confidence_rate, 1.0, 0.0)
        weighted_image = image * mask
        return weighted_image

# --- NEW: Triplet Data Generator ---
class TripletGenerator(tf.keras.utils.Sequence):
    def __init__(self, image_dir, confidence_map_dir, batch_size, img_size, shuffle=True):
        self.image_dir = image_dir
        self.confidence_map_dir = confidence_map_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.shuffle = shuffle
        
        self.class_to_files = self._get_class_files()
        self.class_names = list(self.class_to_files.keys())
        self.on_epoch_end()

    def _get_class_files(self):
        class_to_files = {}
        for class_name in sorted(os.listdir(self.image_dir)):
            class_dir = os.path.join(self.image_dir, class_name)
            if os.path.isdir(class_dir):
                class_to_files[class_name] = [os.path.join(class_name, f) for f in os.listdir(class_dir)]
        return class_to_files

    def __len__(self):
        # An approximation of steps per epoch
        return int(np.floor(sum(len(files) for files in self.class_to_files.values()) / self.batch_size))

    def __getitem__(self, index):
        anchors_img = []
        positives_img = []
        negatives_img = []
        
        anchors_conf = []
        positives_conf = []
        negatives_conf = []

        for _ in range(self.batch_size):
            # 1. Pick anchor and positive from the same class
            anchor_class = random.choice(self.class_names)
            anchor_path, positive_path = random.sample(self.class_to_files[anchor_class], 2)
            
            # 2. Pick negative from a different class
            negative_class = random.choice([c for c in self.class_names if c != anchor_class])
            negative_path = random.choice(self.class_to_files[negative_class])

            # 3. Load images and confidence maps
            anchor_img, anchor_conf_map = self._load_image_and_confidence(anchor_path)
            positive_img, positive_conf_map = self._load_image_and_confidence(positive_path)
            negative_img, negative_conf_map = self._load_image_and_confidence(negative_path)
            
            anchors_img.append(anchor_img)
            positives_img.append(positive_img)
            negatives_img.append(negative_img)
            
            anchors_conf.append(anchor_conf_map)
            positives_conf.append(positive_conf_map)
            negatives_conf.append(negative_conf_map)

        # The 'y' is a dummy value, as the loss is calculated from the three inputs
        return [np.array(anchors_img), np.array(positives_img), np.array(negatives_img),
                np.array(anchors_conf), np.array(positives_conf), np.array(negatives_conf)], np.zeros(self.batch_size)

    def _load_image_and_confidence(self, file_path):
        image_path = os.path.join(self.image_dir, file_path)
        _, filename = os.path.split(file_path)
        confidence_path = os.path.join(self.confidence_map_dir, filename)

        image = np.array(Image.open(image_path).resize(self.img_size)) / 255.0
        confidence = np.array(Image.open(confidence_path).resize(self.img_size)) / 255.0
        
        return image, confidence

    def on_epoch_end(self):
        if self.shuffle:
            # Re-shuffle file lists within each class
            for class_name in self.class_to_files:
                random.shuffle(self.class_to_files[class_name])

# --- NEW: Function to create the embedding model ---
def create_embedding_model(img_size, embedding_dim):
    image_input = Input(shape=(img_size[0], img_size[1], 1), name="image_input")
    confidence_input = Input(shape=(img_size[0], img_size[1], 1), name="confidence_input")

    x = ConfidenceDropout()([image_input, confidence_input])

    x = layers.Conv2D(256, (5, 5), activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(256, (2, 2), activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="swish")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.25)(x)
    
    # --- MODIFICATION: Output is a 128-dim embedding vector ---
    embedding = layers.Dense(embedding_dim, activation=None)(x)
    # L2 Normalization is crucial for stable training with triplet loss
    embedding = layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1), name="embedding_vector")(embedding)

    return Model(inputs=[image_input, confidence_input], outputs=embedding, name="embedding_model")

# --- NEW: Triplet Loss Function ---
def triplet_loss(y_true, y_pred, alpha=0.2):
    anchor, positive, negative = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:384]
    
    # Calculate squared Euclidean distances
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=-1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=-1)
    
    # Calculate the loss
    basic_loss = pos_dist - neg_dist + alpha
    loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0))
    
    return loss

# --- Main script ---
start = time.time()

# 1. Create the data generator
train_generator = TripletGenerator(
    image_dir=config["dataset_root"],
    confidence_map_dir=config["confidence_map_dir"],
    batch_size=config["batch_size"],
    img_size=config["img_size"]
)

# 2. Create the core embedding model
embedding_model = create_embedding_model(config["img_size"], config["embedding_dim"])
embedding_model.summary()

# 3. Create the training model with triplet inputs
anchor_img_input = Input(shape=(config["img_size"][0], config["img_size"][1], 1), name="anchor_image")
positive_img_input = Input(shape=(config["img_size"][0], config["img_size"][1], 1), name="positive_image")
negative_img_input = Input(shape=(config["img_size"][0], config["img_size"][1], 1), name="negative_image")

anchor_conf_input = Input(shape=(config["img_size"][0], config["img_size"][1], 1), name="anchor_confidence")
positive_conf_input = Input(shape=(config["img_size"][0], config["img_size"][1], 1), name="positive_confidence")
negative_conf_input = Input(shape=(config["img_size"][0], config["img_size"][1], 1), name="negative_confidence")

anchor_embedding = embedding_model([anchor_img_input, anchor_conf_input])
positive_embedding = embedding_model([positive_img_input, positive_conf_input])
negative_embedding = embedding_model([negative_img_input, negative_conf_input])

# Concatenate embeddings to pass to the loss function as a single tensor
merged_output = layers.Concatenate(axis=-1)([anchor_embedding, positive_embedding, negative_embedding])

# The model for training
training_model = Model(
    inputs=[anchor_img_input, positive_img_input, negative_img_input, 
            anchor_conf_input, positive_conf_input, negative_conf_input],
    outputs=merged_output
)

# 4. Compile the training model with the custom loss
training_model.compile(
    optimizer=Adam(learning_rate=config["learning_rate"]),
    loss=lambda y_true, y_pred: triplet_loss(y_true, y_pred, alpha=config["triplet_alpha"])
)

# 5. Train the model
print("Starting training with triplet loss...")
history = training_model.fit(
    train_generator,
    epochs=config["epochs"],
    verbose=1
)

# 6. Save the core embedding model (this is what you use for inference)
embedding_model.save(os.path.join(output_dir, config["model_name"]))
print(f"Embedding model saved to {os.path.join(output_dir, config['model_name'])}")

# --- NEW: Evaluation Method ---
# We can't use classification_report directly. Instead, we generate embeddings
# for the test set and use a k-NN classifier to measure class separability.

print("Evaluating the embedding model...")

# Helper to load all test data
def load_test_data(test_dir, confidence_dir, img_size):
    images, conf_maps, labels = [], [], []
    class_indices = {name: i for i, name in enumerate(sorted(os.listdir(test_dir)))}
    
    for class_name, class_idx in class_indices.items():
        class_path = os.path.join(test_dir, class_name)
        if os.path.isdir(class_path):
            for filename in os.listdir(class_path):
                img_path = os.path.join(class_path, filename)
                conf_path = os.path.join(confidence_dir, filename)
                
                img = np.array(Image.open(img_path).resize(img_size)) / 255.0
                conf = np.array(Image.open(conf_path).resize(img_size)) / 255.0

                images.append(img)
                conf_maps.append(conf)
                labels.append(class_idx)
                
    return np.array(images), np.array(conf_maps), np.array(labels)

train_images, train_conf, train_labels = load_test_data(config["dataset_root"], config["confidence_map_dir"], config["img_size"])
test_images, test_conf, test_labels = load_test_data(config["test_data_dir"], config["confidence_map_dir"], config["img_size"])

# Generate embeddings
print("Generating embeddings for training data...")
train_embeddings = embedding_model.predict([train_images, train_conf])
print("Generating embeddings for test data...")
test_embeddings = embedding_model.predict([test_images, test_conf])

# Use k-NN to classify the test embeddings
print("Classifying test embeddings using k-NN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(train_embeddings, train_labels)
predictions = knn.predict(test_embeddings)

# Calculate accuracy
accuracy = accuracy_score(test_labels, predictions)
print(f"\nK-NN classification accuracy on test embeddings: {accuracy:.4f}")

# Save the accuracy result
with open(os.path.join(output_dir, "knn_accuracy_report.txt"), "w") as f:
    f.write(f"K-NN classification accuracy: {accuracy:.4f}\n")
    f.write("This accuracy measures how well the learned embeddings separate the classes in the test set.\n")

# --- Plotting the training loss ---
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Triplet Loss')
plt.title('Model Triplet Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.savefig(os.path.join(output_dir, "training_loss_curve.png"))
plt.clf()


end = time.time()
print(f"Training and evaluation completed. Total time: {end - start:.2f} seconds")