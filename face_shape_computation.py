import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import os

# Define file paths
csv_path = "data/facial_landmarks.csv"
output_path = "data/face_shapes.csv"

# Ensure the output directory exists
os.makedirs(os.path.dirname(output_path), exist_ok=True)

# Define batch size
BATCH_SIZE = 5000  

# Function to determine face shape using only X and Y coordinates
def determine_face_shape(row):
    try:
        # Extract only X and Y coordinates (ignore Z values)
        required_landmarks = [
            "X_152", "Y_152",  # Chin
            "X_10", "Y_10",    # Forehead
            "X_0", "Y_0",      # Left jaw
            "X_16", "Y_16",    # Right jaw
            "X_234", "Y_234",  # Left cheekbone
            "X_454", "Y_454"   # Right cheekbone
        ]

        # Ensure all required landmarks are available
        if row[required_landmarks].isnull().any():
            return "Unknown"

        chin = np.array([row["X_152"], row["Y_152"]])
        forehead = np.array([row["X_10"], row["Y_10"]])
        jaw_left = np.array([row["X_0"], row["Y_0"]])
        jaw_right = np.array([row["X_16"], row["Y_16"]])
        cheek_left = np.array([row["X_234"], row["Y_234"]])
        cheek_right = np.array([row["X_454"], row["Y_454"]])

        # Compute distances using X and Y coordinates only
        face_length = euclidean(forehead, chin)
        jaw_width = euclidean(jaw_left, jaw_right)
        cheekbone_width = euclidean(cheek_left, cheek_right)

        jaw_to_face_ratio = jaw_width / face_length
        cheek_to_face_ratio = cheekbone_width / face_length

        # Determine face shape based on ratios
        if cheek_to_face_ratio > 0.9 and jaw_to_face_ratio > 0.8:
            return "Round"
        elif jaw_to_face_ratio > 0.75 and cheek_to_face_ratio > 0.85:
            return "Square"
        elif cheek_to_face_ratio > 0.85 and jaw_to_face_ratio < 0.7:
            return "Diamond"
        elif jaw_to_face_ratio < 0.7:
            return "Oval"
        else:
            return "Heart"

    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return "Unknown"

# Read CSV in batches
chunk_iter = pd.read_csv(csv_path, chunksize=BATCH_SIZE)

# Process each batch
for chunk_idx, chunk in enumerate(chunk_iter):
    print(f"Processing batch {chunk_idx + 1}...")

    # Pivot to get one row per image
    df_pivot = chunk.pivot(index="Image_ID", columns="Landmark_ID", values=["X", "Y"]).reset_index()
    df_pivot.columns = ['_'.join(map(str, col)).strip() for col in df_pivot.columns]

    # Apply function to determine face shape
    df_pivot["Face_Shape"] = df_pivot.apply(determine_face_shape, axis=1)

    # Save results (append after first batch)
    mode = 'w' if chunk_idx == 0 else 'a'
    header = True if chunk_idx == 0 else False
    df_pivot[["Image_ID_", "Face_Shape"]].to_csv(output_path, index=False, mode=mode, header=header)

    print(f"Batch {chunk_idx + 1} saved.")

print("All batches processed and saved to data/face_shapes.csv.")
