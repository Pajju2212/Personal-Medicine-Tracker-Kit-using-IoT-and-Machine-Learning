import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import time

# Function to load images from a directory and assign a label
def load_images_from_folder(folder, label):
    images = []
    labels = []
    try:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            img = cv2.imread(file_path)
            if img is not None:
                img = cv2.resize(img, (224, 224))  # Resize images
                images.append(img)
                labels.append(label)
            else:
                print(f"Warning: Could not read image {file_path}")
    except Exception as e:
        print(f"Error: {e}")
    return np.array(images), np.array(labels)

# Load datasets from two folders
taking_pill_folder = r'C:\Users\Aishwarya Arahunasi\OneDrive\Desktop\mini_photos\pill_taking'
not_taking_pill_folder = r'C:\Users\Aishwarya Arahunasi\OneDrive\Desktop\mini_photos\pill_not_taking'
images_taking_pill, labels_taking_pill = load_images_from_folder(taking_pill_folder, 1)
images_not_taking_pill, labels_not_taking_pill = load_images_from_folder(not_taking_pill_folder, 0)

# Combine the datasets
images = np.concatenate((images_taking_pill, images_not_taking_pill), axis=0)
labels = np.concatenate((labels_taking_pill, labels_not_taking_pill), axis=0)

# Normalize pixel values
images = images / 255.0

# Convert labels to categorical
label_binarizer = LabelBinarizer()
labels_categorical = label_binarizer.fit_transform(labels).flatten()

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.3, random_state=42)

# Load the VGG16 model with pre-trained ImageNet weights, excluding the top layer
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
model = Model(inputs=base_model.input, outputs=base_model.output)

# Extract features
X_train_features = model.predict(X_train)
X_test_features = model.predict(X_test)

# Flatten the features
X_train_features_flat = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_features_flat = X_test_features.reshape(X_test_features.shape[0], -1)

# Train an SVM classifier
svm = SVC(kernel='linear', probability=True)
svm.fit(X_train_features_flat, y_train)

# Predict on the test set
y_pred = svm.predict(X_test_features_flat)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(svm, 'svm_pill_classification_model.joblib')

# Load the trained SVM model
svm = joblib.load('svm_pill_classification_model.joblib')

# Function to preprocess the frame
def preprocess_frame(frame):
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Replace this with your ESP32-CAM stream URL
url = "http://192.168.20.212:81/stream"  # Change to your ESP32-CAM IP address

# Open the video stream
cap = cv2.VideoCapture(url)

# Variables to control frame processing every second
last_time = time.time()
fps_interval = 1  # Process every 1 second

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image from ESP32-CAM")
        break

    # Get the current time to ensure we process frames every second
    current_time = time.time()

    if current_time - last_time >= fps_interval:
        last_time = current_time

        # Preprocess the frame
        preprocessed_frame = preprocess_frame(frame)

        # Extract features using the CNN
        features = model.predict(preprocessed_frame)
        features_flat = features.reshape(1, -1)

        # Predict using the SVM
        prediction = svm.predict(features_flat)
        prediction_text = 'Taking pill' if prediction[0] == 1 else 'Not taking pill'

        # Print the prediction to the console
        print(f"Prediction: {prediction_text}")

        # Display the prediction on the frame
        cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display the frame (optional, useful for visualizing the stream)
    cv2.imshow('ESP32-CAM Stream', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
