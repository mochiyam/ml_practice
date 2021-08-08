
# Using mnist.npz from the Coursera Jupyter Notebook to classify 10 different type of clothes.

import tensorflow as tf
from os import path, getcwd, chdir


path = f"{getcwd()}/../tmp2/mnist.npz"

def train_mnist():
    # Please write your code only where you are indicated.
    # please do not remove # model fitting inline comments.
    
    class myCallbacks(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('acc') is not None and logs.get('acc') > 0.99):
                print("\nReached 99% accuracy so cancelling training!")
                self.model.stop_training = True
    
    callbacks = myCallbacks()
    mnist = tf.keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data(path=path)
    x_train = x_train / 225.0
    x_test = x_test / 225.0
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    # model fitting
    history = model.fit(x_train, y_train, epochs=10, callbacks=[callbacks])
    # model fitting
    return history.epoch, history.history['acc'][-1]

train_mnist()