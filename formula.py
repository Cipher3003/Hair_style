import pandas as pd
import numpy as np
from scipy.spatial.distance import euclidean
import os

# Load Extracted Landmarks
df = pd.read_csv("published_dataset/extracted_landmarks.csv")

# Function to compute ratios
def compute_ratios(row):
    # Extract X, Y coordinates (ignoring Z)
    chin = np.array([row["CHIN_X"], row["CHIN_Y"]])
    forehead = np.array([row["FOREHEAD_X"], row["FOREHEAD_Y"]])
    jaw_left = np.array([row["JAW_START_X"], row["JAW_START_Y"]])
    jaw_right = np.array([row["JAW_END_X"], row["JAW_END_Y"]])
    cheek_left = np.array([row["LEFT_CHEEK_X"], row["LEFT_CHEEK_Y"]])
    cheek_right = np.array([row["RIGHT_CHEEK_X"], row["RIGHT_CHEEK_Y"]])

    # Compute distances
    face_length = euclidean(forehead, chin)
    jaw_width = euclidean(jaw_left, jaw_right)
    cheekbone_width = euclidean(cheek_left, cheek_right)

    # Compute ratios
    row["Jaw_to_Face_Ratio"] = jaw_width / face_length
    row["Cheek_to_Face_Ratio"] = cheekbone_width / face_length
    return row

# Apply function to compute ratios
df = df.apply(compute_ratios, axis=1)

# Compute Average Ratios for Each Face Shape
shape_ratios = df.groupby("Face_Shape")[["Jaw_to_Face_Ratio", "Cheek_to_Face_Ratio"]].mean()

# Save Computed Ratios
shape_ratios.to_csv("published_dataset/face_shape_ratios.csv")
print("Classification formula saved to face_shape_ratios.csv")
