# load libraries
import numpy as np
import os
import tensorflow as tf
import keras
from tensorflow.keras.callbacks import EarlyStopping

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

class_names = train_ds.class_names

# configure the dataset
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)
"""
# data augmentation
data_augmentation = keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal',
                                                          input_shape=(image_height,
                                                                       image_width,
                                                                       3)),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
    tf.keras.layers.experimental.preprocessing.RandomZoom(0.1)
    ])
"""
#create the model
num_classes = 2
model = tf.keras.Sequential([
    #data_augmentation,
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape = (image_height, image_width, 3)),
    tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes)
    ])

model.summary()

# compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )

# early stop and checkpoints
checkpoint_filepath = "G:/rauf/STEPBYSTEP/Projects2/vision/Image Classification/checkpoint_path"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = checkpoint_filepath,
    save_weights_only=False,
    monitor='loss',
    mode='min',
    save_best_only=True)

callbacks = [EarlyStopping(patience=2),
             model_checkpoint_callback]

# train the model
epochs=15
history = model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=epochs,
    callbacks=callbacks)

# predict random image
image_path = "G:/rauf/STEPBYSTEP/Data2/dog_cat/test/cat/6.jpg"
img = tf.keras.preprocessing.image.load_img(
    image_path,
    target_size=(image_height, image_width))
image_array = tf.keras.preprocessing.image.img_to_array(img)
image_array = tf.expand_dims(image_array, 0)

prediction = model.predict(image_array)
score = tf.nn.softmax(prediction[0])
print("This image most likely belong to {}, with the accuracy {:.2f} persent"
      .format(class_names[np.argmax(score)], 100*np.max(score)))

# well done our random image predicted correctly with the 98% accuracy
