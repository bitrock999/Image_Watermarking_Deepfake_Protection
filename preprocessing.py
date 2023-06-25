import cv2
import os
import json

# Load settings from setting.json
with open('setting.json') as file:
    settings = json.load(file)

data_path = settings['global_settings']['data_path_KPM']
results_path = settings['global_settings']['results_path_FasilkomUnsri']

# Create results directory if it doesn't exist
if not os.path.exists(results_path):
    os.makedirs(results_path)

# Function to upscale and crop images
def enhance_image(image):
    # Upscale the image by 2x
    upscaled_image = cv2.resize(image, None, fx=2, fy=2)

    # Detect faces in the upscaled image using OpenCV's face detection algorithm
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Crop the detected faces to size 128x128
    cropped_faces = []
    for (x, y, w, h) in faces:
        cropped_face = upscaled_image[y:y+h, x:x+w]
        cropped_face = cv2.resize(cropped_face, (128, 128))
        cropped_faces.append(cropped_face)

    return cropped_faces

# Iterate over the images in the dataset
for filename in os.listdir(data_path):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Load the image
        image_path = os.path.join(data_path, filename)
        image = cv2.imread(image_path)

        # Enhance the image
        enhanced_images = enhance_image(image)

        # Save the enhanced images
        for i, enhanced_image in enumerate(enhanced_images):
            result_filename = f"{os.path.splitext(filename)[0]}_{i}.jpg"
            result_path = os.path.join(results_path, result_filename)
            cv2.imwrite(result_path, enhanced_image)

print("Image enhancement complete.")
