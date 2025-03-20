import cv2

# Use raw string (r"")
image_path = r"C:\Users\sai\OneDrive\Documents\NCU\Hair style\data\img_align_celeba\img_align_celeba\065783.jpg"
image = cv2.imread(image_path)

if image is None:
    print("Error: Failed to load image. Check the file path or image integrity.")
else:
    print("Size:", image.shape)  # (height, width, channels)
    if image.shape[-1] == 3:
        print("Mode: BGR (OpenCV loads images in BGR by default)")
