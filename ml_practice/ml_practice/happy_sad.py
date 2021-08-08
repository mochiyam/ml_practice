
# Use happy or sad dataset that contains 80 images (40 happy and 40 sad)
# Stop training once it hits 100% accuracy on the images, which cancels training
#  upon hitting training accuracy of > .999
# Assumption : 150 x 150 pixel in the implementation

import tensorflow as tf
import os
import zipfile
from os import path, getcwd, chdir

path = f"{getcwd()}/../tmp2/happy-or-sad.zip"

zip_ref = zipfile.ZipFile(path, 'r')
zip_ref.extractall("/tmp/h-or-s")
zip_ref.close()

def train_happy_sad_model():
    DESIRED_ACCURACY = 0.999
    
    # Callback to end training once it hits 99.9%
    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') > DESIRED_ACCURACY):
                print("\nReached 99.9% accuracy so cancelling training")
                self.model.stop_training = True

    callbacks = myCallbacks()

    # Creating a model using convolutional layers
    mode = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    from tensorflow.keras.optimizers import RMSprop

    # Configures the model for training
    model.compile(loss='binary_crossentropy',
                 optimizer=RMSprop(lr=0.001),
                 metrics=['accuracy'])

    # Data Preprocessing
    # Set up data generator that will read pictures from the source 

    train_datagen = ImageDataGenerator(rescale=1/255)

    # Set target_size of 150 x 150
    train_generator = train.datagen.flow_from_directory(
        '/tmp/h-or-s',
        target_size=(150, 150),
        batch_size=10,
        class_mode='binary')

    # Fits the model on data yielded batch-by-batch
    model.fit_generator(
        train_generator, 
        steps_per_epoch=8,
        epochs=15,
        callbacks=[callbacks])

      # Model fitting
    return history.history['acc'][-1]

train_happy_sad_model()