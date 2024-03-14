import cv2
import numpy as np
import os
import glob
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, recall_score

# Function to load and preprocess images
def load_data(path):
    images = []
    labels = []

    # Load ripe tomatoes images
    for file in glob.glob(os.path.join(path, 'Images', 'Riped tomato_*.jpeg')):
        img = cv2.imread(file)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(1)  # label 1 for ripe tomatoes

    # Load unripe tomatoes images
    for file in glob.glob(os.path.join(path, 'Images', 'unriped tomato_*.jpeg')):
        img = cv2.imread(file)
        img = cv2.resize(img, (224, 224))
        images.append(img)
        labels.append(0)  # label 0 for unripe tomatoes

    # Convert to numpy arrays and normalize images to 0-1 range
    images = np.array(images) / 255.0
    labels = np.array(labels)

    # Split data into training and testing sets
    train_data, test_data, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

    return train_data, test_data, train_labels, test_labels

# Function to create the model
def create_model():
    model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Function to train the model
def train_model(model, train_data, train_labels, test_data, test_labels):
    model.fit(train_data, train_labels, epochs=10, validation_data=(test_data, test_labels))
    
    # Save the model
    model.save('tomato_model.h5')
    
    return model

# Function to evaluate the model
def evaluate_model(model, test_data, test_labels):
    loss, accuracy = model.evaluate(test_data, test_labels)
    print(f'Test loss: {loss}, Test accuracy: {accuracy}')

    # Predict labels for the test data
    predicted_labels = model.predict(test_data)
    predicted_labels = (predicted_labels > 0.5).astype(np.int32)

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(test_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate precision and recall
    precision = precision_score(test_labels, predicted_labels)
    recall = recall_score(test_labels, predicted_labels)

    print("Precision:", precision)
    print("Recall:", recall)

    # Display results in tabular format
    print("\nResults:")
    print("--------------")
    print("| Metric     | Value   |")
    print("|------------|---------|")
    print(f"| Accuracy   | {accuracy:.4f} |")
    print(f"| Precision  | {precision:.4f} |")
    print(f"| Recall     | {recall:.4f} |")
    print("--------------")

# Load data
train_data, test_data, train_labels, test_labels = load_data('dataset')

# Create model
model = create_model()

# Train model
model = train_model(model, train_data, train_labels, test_data, test_labels)

# Evaluate model
evaluate_model(model, test_data, test_labels)
