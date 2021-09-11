# load libraries
import numpy as np
import os
import tensorflow as tf

#load dataset
train_data_path = "G:/rauf/STEPBYSTEP/Data2/dog_cat/train"

batch_size = 2
image_height = 180
image_width = 180

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_path,
    validation_split=0.2,
    subset='training',
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size)

validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_data_path,
    validation_split=0.2,
    subset='validation',
    seed=123,
    image_size=(image_height, image_width),
    batch_size=batch_size)
