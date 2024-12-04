import argparse
import pickle
from pathlib import Path
import face_recognition
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import cv2

# Global Constants
DEFAULT_ENCODINGS_PATH = Path("output/encodings.pkl")
# Color configuration for Square and text
BOUNDING_BOX_COLOR = (0, 255, 0)
TEXT_COLOR = (255, 255, 255)

# Function to train model with known images from the specified directory and save the face encodings
def train_model(known_images_dir: str, encodings_path: Path = DEFAULT_ENCODINGS_PATH):
    known_encodings = []
    known_names = []

    # Iterate through each image in the training directory
    for root, _, files in os.walk(known_images_dir):
        for file in files:
            # Takes any file ended with specific image format
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root, file)
                # Use folder name as the label for the image
                name = os.path.basename(root)  
                # Load the image file from the path
                image = face_recognition.load_image_file(image_path)
                # Compute the facial encodings 
                encodings = face_recognition.face_encodings(image)
                # If facial encodings are found in the image then append encoding and name
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(name)

    # Save the encodings to a file
    data = {"encodings": known_encodings, "names": known_names}
    os.makedirs(encodings_path.parent, exist_ok=True)
    with open(encodings_path, "wb") as f:
        pickle.dump(data, f)
    print(f"Training completed and saved to: {encodings_path}")

# function that recognize faces from the weebcam
def recognize_faces(encodings_location=DEFAULT_ENCODINGS_PATH, model="hog"):
    # Load the trained encodings
    with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)

    # Initialize the webcam
    video_capture = cv2.VideoCapture(0)
    
    # Handling erros if came is not accessible
    if not video_capture.isOpened():
        print("Error: Could not access the camera.")
        return

    print("Press 'q' to exit the webcam view.")

    while True:
        # Capture a single frame
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to grab frame.")
            break
        
        # Convert the frame from BGR (OpenCV) to RGB (face_recognition)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find all face locations and face encodings in the current frame
        face_locations = face_recognition.face_locations(rgb_frame, model=model)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Compare face encodings with the known encodings
            distances = face_recognition.face_distance(loaded_encodings["encodings"], face_encoding)
            name = "Unknown"
            # If encodings of the face and the stored encodings are less than 0.6 distance then label it before displaying the frame
            # Its possible to modify the distance, 0.6 according to my research, normally works.
            if len(distances) > 0 and min(distances) < 0.6:
                best_match_index = distances.argmin()
                name = loaded_encodings["names"][best_match_index]
            
            # Draw bounding box and label on the frame
            cv2.rectangle(frame, (left, top), (right, bottom), BOUNDING_BOX_COLOR, 2)
            cv2.rectangle(frame, (left, bottom - 20), (right, bottom), BOUNDING_BOX_COLOR, cv2.FILLED)
            cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
        
        # Display the frame with annotations
        cv2.imshow('Face Recognition', frame)
        
        # Exit the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the webcam and close OpenCV windows
    video_capture.release()
    cv2.destroyAllWindows()

# Function to validate the model using images from the val directiory and print performance metrics for learning purposes
def validate_model(val_images_dir: str, encodings_path: Path = DEFAULT_ENCODINGS_PATH, model: str = "hog"):
    # We use the hog model for fast responses
    # Load the trained encodings
    with open(encodings_path, "rb") as f:
        loaded_encodings = pickle.load(f)

    true_labels = []
    predicted_labels = []

    # Iterate through each image in the validation directory
    for root, _, files in os.walk(val_images_dir):
        for file in files:
            if file.lower().endswith(('jpg', 'jpeg', 'png')):
                image_path = os.path.join(root, file)
                 # Folder name as ground-truth label
                true_label = os.path.basename(root) 

                # Load the input image and extract face encodings
                input_image = face_recognition.load_image_file(image_path)
                input_face_locations = face_recognition.face_locations(input_image, model=model)
                input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

                for unknown_encoding in input_face_encodings:
                    # Calculate distances and find the best match
                    # barely the same implementation we used before for the face recognition
                    distances = face_recognition.face_distance(loaded_encodings["encodings"], unknown_encoding)
                    if len(distances) > 0 and min(distances) < 0.6:
                        best_match_index = distances.argmin()
                        predicted_label = loaded_encodings["names"][best_match_index]
                    else:
                        predicted_label = "Unknown"

                    true_labels.append(true_label)
                    predicted_labels.append(predicted_label)

    # Calculate and print validation metrics from sklearn
    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    recall = recall_score(true_labels, predicted_labels, average='weighted', zero_division=0)
    f1 = f1_score(true_labels, predicted_labels, average='weighted', zero_division=0)

    print(f"Validation Metrics:\nAccuracy: {accuracy:.2f}\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1-score: {f1:.2f}")

# Argument parser setup
parser = argparse.ArgumentParser(description="Face recognition script")
parser.add_argument("--train", help="Train with a directory of known faces", metavar="DIR")
parser.add_argument("--validate", help="Validate the model with a directory of validation images", metavar="DIR")
parser.add_argument("--model", default="hog", choices=["hog", "cnn"], help="Face detection model to use (hog or cnn)")

# Parse arguments and execute corresponding functions
args = parser.parse_args()
# In a real implementation we should modify this
recognize_faces()

if args.train:
    train_model(args.train)

if args.validate:
    validate_model(args.validate, model=args.model)