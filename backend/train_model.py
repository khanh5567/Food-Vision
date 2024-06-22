#pip install h5py pandas matplotlib opencv-python keras tensorflow
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications import VGG16
from keras.preprocessing.image import load_img, img_to_array
import warnings
warnings.filterwarnings('ignore')

models_filename = 'v8_vgg16_model_1.h5'
image_dir = 'food101/images/'
image_size = (224, 224)
batch_size = 16
epochs = 80

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
rescale = 1./255,
horizontal_flip = False,
fill_mode = "nearest",
zoom_range = 0,
width_shift_range = 0,
height_shift_range=0,
rotation_range=0)

train_generator = train_datagen.flow_from_directory(
image_dir,
target_size = (image_size[0], image_size[1]),
batch_size = batch_size,
class_mode = "categorical")

num_of_classes = len(train_generator.class_indices)

model = VGG16(weights=None, include_top=False, input_shape=(image_size[0], image_size[1], 3))

#Adding custom Layers
x = model.output
x = Flatten()(x)
x = Dense(101*2, activation="relu")(x)
x = Dense(101*2, activation="relu")(x)
predictions = Dense(101, activation="softmax")(x)
model_final = Model(inputs=model.input, outputs=predictions)
model_final.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
model_final.load_weights(models_filename)

model_final.save('my_model.keras')