import time
import numpy as np

import os
import tensorflow as tf
import PIL
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt
import scipy

import cv2

# Read the input image
input_settings = 0
input_name = 'input/input' + str(input_settings) + '.jpg'

img = cv2.imread(input_name)

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
    print()
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
            cv2.rectangle(gray_buffer, (x, y), (x + w, y + h),
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

angle_best = angles[np.argmin(vecs)]
print("Best angle: " + str(angle_best))

trans = cv2.getRotationMatrix2D(center, angle_best, scale)
gray = cv2.warpAffine(gray, trans, (width, height))

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 4)
trans_colored = cv2.getRotationMatrix2D(center, angle_best, scale)
if len(faces) == 0:  # Rotate 180 degrees in case no face was detected
    trans = cv2.getRotationMatrix2D(center, 180, scale)
    gray = cv2.warpAffine(gray, trans, (width, height))
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    trans_colored = cv2.getRotationMatrix2D(center, angle_best + 180, scale)
img = cv2.warpAffine(img, trans_colored, (width, height))

# Draw rectangle around the faces and crop the faces
for (x, y, w, h) in faces:
    cv2.rectangle(gray, (x, y), (x + w, y + h), (255, 0, 0), 2)
    faces = gray[y:y + h, x:x + w]
    cv2.imwrite('data/data/faces.jpg', faces)
    break

# Configures GPU memory usage
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

# Set Path
test_dir = os.path.join("data")

# Get FER-2013 model and weight
model = tf.keras.models.load_model("datamodel/ferNet.h5")
model.load_weights('datamodel/fernet_bestweight.h5')

# Make probability model
probability_model = tf.keras.Sequential([model,
                                         tf.keras.layers.Softmax()])

# Input image preprocessor
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_set = test_datagen.flow_from_directory(test_dir,
                                            batch_size=64,
                                            target_size=(48, 48),
                                            shuffle=True,
                                            color_mode='grayscale',
                                            class_mode='categorical')

# Define class names and emojis
classname = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
emoji = [
    "emojis/angry.png", "emojis/disgust.png", "emojis/fear.png", "emojis/happy.png",
    "emojis/neutral.png", "emojis/sad.png", "emojis/surprise.png"
]

# Count total images
totalcount = 0
for totalfilenum in os.listdir(test_dir + "/data"):
    totalcount += 1

# Predict images and paste corresponding emoji
if (totalcount != 0):
    predictions = probability_model.predict(test_set)
    class_num = int(np.argmax(predictions[0]))

    background_undef = Image.open(input_name)
    background = Image.new("RGBA", background_undef.size)  # Convert grayscale to rgb
    background.paste(background_undef)
    foreground = Image.open(emoji[class_num])
    foreground = foreground.rotate(360 - angle_best, resample=Image.BICUBIC)
    foreground = foreground.resize((int(h * 1.1), int(h * 1.1)))
    width_emoji, height_emoji = foreground.size

    # For testing purposes
    # foreground = Image.open("emojis/test.png")

    # Calculate the actual face position
    x = int(x + w / 2)
    y = int(y + h / 2)

    import math

    x_center = int(width / 2)
    y_center = int(height / 2)
    sin = math.sin(math.radians(360 - angle_best))
    cos = math.cos(math.radians(360 - angle_best))

    x_actual = int((x - x_center) * cos - ((height - y) - y_center) * sin + x_center)
    y_actual = int((x - x_center) * sin + ((height - y) - y_center) * cos + y_center)

    # Paste
    background.paste(foreground, (x_actual - int(width_emoji / 2), (height - y_actual - int(width_emoji / 2))), foreground.convert('RGBA'))

    # For debug
    radius = int(math.sqrt((x - x_center) ** 2 + (y - y_center) ** 2))
    cv2.circle(gray, center, radius, (255, 255, 255), thickness=1, lineType=cv2.LINE_8, shift=0)
    cv2.line(gray, (x_center, y_center), (x_actual, (height - y_actual)), (255, 255, 255))
    cv2.line(gray, (x_center, y_center), (x, y), (255, 255, 255))

    background.show()
    print("Best angle: " + str(angle_best))
    print("Prediction: " + str(classname[class_num]))
    cv2.imshow("face", gray)
    cv2.waitKey()

else:
    print("No image was found in directory")
