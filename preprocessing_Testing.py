import cv2

image_A = cv2.imread('data/KPM_Fasilkom_UNSRI_2018/Sistem_Informasi_bilingual/09031381823104.jpg')
image_B = cv2.imread('data/KPM_Fasilkom_UNSRI_2018/Sistem_Informasi_bilingual/09031381823105.jpg')

# Upscale the images by 2x using resize()
upscaled_A = cv2.resize(image_A, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
upscaled_B = cv2.resize(image_B, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)

# Save the upscaled images
cv2.imwrite('A2.jpg', upscaled_A)
cv2.imwrite('B2.jpg', upscaled_B)

# Function to detect and crop the middle face
def detect_and_crop_middle_face(image):
    # Convert image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform face detection using Haar cascades
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Find the middle face
    middle_index = len(faces) // 2
    (x, y, w, h) = faces[middle_index]

    # Crop the middle face
    cropped_face = image[y:y+h, x:x+w]
    

    return (x, y, w, h), cropped_face

# Perform face detection and cropping on image A2
(x_A, y_A, w_A, h_A), cropped_A = detect_and_crop_middle_face(upscaled_A)

# Perform face detection and cropping on image B2
(x_B, y_B, w_B, h_B), cropped_B = detect_and_crop_middle_face(upscaled_B)

# Resize cropped_B to match the size of cropped_A
cropped_A = cv2.resize(cropped_A, (w_A, h_A))
cv2.imwrite('cropped_face A.jpg', cropped_A)

cropped_B = cv2.resize(cropped_B, (w_B, h_B))
cv2.imwrite('cropped_face B.jpg', cropped_B)

