import cv2
from mtcnn import MTCNN
import matplotlib.pyplot as plt

# Initialize MTCNN detector
detector = MTCNN()

# Load image
image_path = "Screenshot 2024-05-31 002252.png"  # Change to your image path
image = cv2.imread(image_path)
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Detect faces
faces = detector.detect_faces(rgb_image)

# Draw bounding boxes
for face in faces:
    x, y, width, height = face['box']
    confidence = face['confidence']
    cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    cv2.putText(image, f'Conf: {confidence:.2f}', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the image
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis("off")
plt.show()
