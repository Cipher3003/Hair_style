import cv2
import mediapipe as mp
import os
import pandas as pd

# MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)

# Landmark IDs
LANDMARKS = {
    "JAW_START": 0,   # Start of jawline
    "JAW_END": 16,     # End of jawline
    "LEFT_CHEEK": 234, # Left cheekbone
    "RIGHT_CHEEK": 454,# Right cheekbone
    "CHIN": 152,       # Chin point
    "FOREHEAD": 10     # Forehead center
}

# Input Dataset Directory
DATASET_DIR = "published_dataset/"
OUTPUT_CSV = "published_dataset/extracted_landmarks.csv"

# Initialize Data Storage
landmark_data = []

# Process Each Category Folder (Face Shape Types)
for face_shape in os.listdir(DATASET_DIR):
    shape_folder = os.path.join(DATASET_DIR, face_shape)
    
    # Ensure it's a folder
    if not os.path.isdir(shape_folder):
        continue

    # Process Each Image
    for image_name in os.listdir(shape_folder):
        image_path = os.path.join(shape_folder, image_name)

        # Read Image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read {image_path}, skipping...")
            continue

        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(image_rgb)

        # Check if a face was detected
        if not results.multi_face_landmarks:
            print(f"No face detected in {image_path}, skipping...")
            continue

        # Extract Landmarks for the First Detected Face
        face_landmarks = results.multi_face_landmarks[0]
        row = {"Image_Name": image_name, "Face_Shape": face_shape}

        for key, landmark_id in LANDMARKS.items():
            landmark = face_landmarks.landmark[landmark_id]
            row[f"{key}_X"] = landmark.x
            row[f"{key}_Y"] = landmark.y
            row[f"{key}_Z"] = landmark.z  # Keeping Z for reference but can be ignored in classification

        landmark_data.append(row)

# Convert to DataFrame and Save
df = pd.DataFrame(landmark_data)
df.to_csv(OUTPUT_CSV, index=False)
print(f"Landmarks extracted and saved to {OUTPUT_CSV}")
