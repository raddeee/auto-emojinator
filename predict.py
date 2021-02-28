# Import libs
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
import matplotlib.pyplot as plt

# Configures GPU memory usage
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)
        print('{} memory growth: {}'.format(device, tf.config.experimental.get_memory_growth(device)))
else:
    print("Not enough GPU hardware devices available")

# Set Path
test_dir = os.path.join("predict")

# Get FER-2013 model and weight
model = tf.keras.models.load_model("ferNet.h5")
model.load_weights('fernet_bestweight.h5')

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

# Define class names
classname = ["angry","disgust","fear","happy","neutral","sad","surprise"]

# Count total images
totalcount = 0
for totalfilenum in os.listdir(test_dir + "/predict"):
    totalcount += 1

# Predict images and plots it
if(totalcount != 0):
    i = 0
    plt.figure(figsize=(totalcount * 5, 5))
    predictions = probability_model.predict(test_set)
    for filenum in os.listdir(test_dir + "/predict"):
        plt.subplot(1, totalcount, i + 1)
        img_predict = load_img((test_dir + "/predict/" + os.listdir(test_dir + "/predict")[i]))
        plt.imshow(img_predict)

        plt.title("Prediction: " + str(classname[int(np.argmax(predictions[i]))]) + " / File " + str(i + 1) + " Name: " +
                  os.listdir(test_dir + "/predict")[i])
        plt.axis('off')
        print(os.listdir(test_dir + "/predict")[i] + "'s predictions: " + str(predictions[i]))
        i += 1
    plt.show()

else:
    print("No image was found in directory")

