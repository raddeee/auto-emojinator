import time
import numpy as np

import cv2

# Read the input image
img = cv2.imread('input/input0.jpg')

# Convert into grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_buffer = gray

# Setting cascade classifier XMLs
eye_cascade = cv2.CascadeClassifier('cascades/haarcascade_eye.xml')
face_cascade = cv2.CascadeClassifier('cascades/haarcascade_frontalface_alt2.xml')

# Setting image properties
height = gray.shape[0]
width = gray.shape[1]
center = (int(width / 2), int(height / 2))

angle = 0
scale = 1.0

# Rotate image and get the best angle
vecs = []
angles = []
for i in range(72):
    time_start = time.time()
    if i != 0:
        angle = i * 5
        trans = cv2.getRotationMatrix2D(center, angle, scale)
        gray_buffer = cv2.warpAffine(gray, trans, (width, height))

    eyes = eye_cascade.detectMultiScale(gray_buffer, 1.1, 4)  # Detect eyes
    eyes_num = len(eyes)
    if eyes_num == 2:
        i2 = 0
        for (x, y, w, h) in eyes:
            cv2.rectangle(img, (x, y), (x + w, y + h),
                          (0, 0, 255), 2)
            if i2 == 0:
                x1 = x
                y1 = y
            else:
                x2 = x
                y2 = y
            i2 += 1
        vec = float(y2 - y1) / float(x2 - x1)  # Calculate vector
        if vec < 0:
            vec = abs(vec)
        print(vec)
        vecs.append(vec)
        angles.append(angle)

print("Best angle: " + str(angles[np.argmin(vecs)]))

trans = cv2.getRotationMatrix2D(center, angles[np.argmin(vecs)], scale)
gray = cv2.warpAffine(gray, trans, (width, height))


# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
trans_colored = cv2.getRotationMatrix2D(center, angles[np.argmin(vecs)], scale)
if len(faces) == 0: # Rotate 180 degrees in case no face was detected
    trans = cv2.getRotationMatrix2D(center, 180, scale)
    gray = cv2.warpAffine(gray, trans, (width, height))
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    trans_colored = cv2.getRotationMatrix2D(center, angles[np.argmin(vecs)] + 180, scale)
img = cv2.warpAffine(img, trans_colored, (width, height))



# Draw rectangle around the faces and crop the faces
for (x, y, w, h) in faces:
    cv2.rectangle(gray, (x, y), (x + w, y + h), (0, 0, 255), 2)
    faces = gray[y:y + h, x:x + w]
    cv2.imshow("face", faces)
    cv2.imwrite('face.jpg', faces)

# Display the output
cv2.imwrite('detcted.jpg', gray)
cv2.imshow('img', gray)

cv2.waitKey()