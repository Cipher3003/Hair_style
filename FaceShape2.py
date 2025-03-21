import cv2
import dlib
import os
# Load pre-trained face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Load the image
image_path = os.path('data','img_align_celeba','img_align_celeba')
# image_path = r"C:\Users\sai\OneDrive\Documents\NCU\Hair style\data\img_align_celeba\img_align_celeba\000038.jpg"  # Change this to your image path
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = detector(gray)

# Loop through detected faces
for face in faces:
    landmarks = predictor(gray, face)  # Predict landmarks

    # Draw landmarks
    for i in range(68):  # dlib provides 68 facial landmarks
        x, y = landmarks.part(i).x, landmarks.part(i).y
        cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Draw circle at each landmark

# Display the result
cv2.imshow("Facial Landmarks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
