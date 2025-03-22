import cv2
import mediapipe as mp
import pandas as pd
import os

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh

# Dynamically get the project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(project_dir, "data", "img_align_celeba", "img_align_celeba")
output_csv = os.path.join(project_dir, "data", "facial_landmarks.csv")

# Define key landmark indices
KEY_LANDMARKS = [0, 16, 234, 454, 152, 10]  # Jaw start, jaw end, left cheek, right cheek, chin, forehead

# Get all image names and split into chunks of 10,000
image_names = sorted(os.listdir(image_folder))  # Sort ensures consistent order
batch_size = 10000  # Number of images per batch
total_batches = len(image_names) // batch_size + (1 if len(image_names) % batch_size != 0 else 0)

# Process images in batches
with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as face_mesh:
    for batch_num in range(total_batches):
        all_landmarks = []  # Store landmarks for the current batch

        # Get the batch images
        batch_images = image_names[batch_num * batch_size: (batch_num + 1) * batch_size]
        
        print(f"Processing batch {batch_num + 1}/{total_batches} ({len(batch_images)} images)")

        for image_name in batch_images:
            image_path = os.path.join(image_folder, image_name)

            # Load Image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Error loading {image_name}")
                continue

            # Convert image to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image
            results = face_mesh.process(rgb_image)

            # If face landmarks are detected
            if results.multi_face_landmarks:
                h, w, _ = image.shape

                for face_landmarks in results.multi_face_landmarks:
                    for idx in KEY_LANDMARKS:  # Only extract key landmarks
                        landmark = face_landmarks.landmark[idx]
                        x, y, z = landmark.x * w, landmark.y * h, landmark.z
                        all_landmarks.append([image_name, idx, x, y, z])

        # Convert batch data to DataFrame
        df = pd.DataFrame(all_landmarks, columns=["Image_ID", "Landmark_ID", "X", "Y", "Z"])

        # Save to CSV (Append after the first batch)
        if batch_num == 0:
            df.to_csv(output_csv, index=False, mode='w', header=True)  # First batch: overwrite and add header
        else:
            df.to_csv(output_csv, index=False, mode='a', header=False)  # Subsequent batches: append without header
        
        print(f"Batch {batch_num + 1} saved to {output_csv}")

print("Processing completed!")
