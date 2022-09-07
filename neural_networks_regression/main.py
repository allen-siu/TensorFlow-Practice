import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Each row of input should be 1 example
X = np.array([[-7, -4, -1, 2, 5, 8, 11, 14]])
y = np.array([[3, 6, 9, 12, 15, 18, 21, 24]])

X = tf.constant(X, dtype=tf.float32)
y = tf.constant(y, dtype=tf.float32)

X = tf.transpose(X)
y = tf.transpose(y)

# plt.scatter(X, y)
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["mae"])

model.fit(X, y, epochs=100)
print(model.predict([17.0]))

# Improve a model by
# Changing these values may not neccessarily improve the model
# - Increasing number of epochs (can never be bad)
# - Using different activation funcs
# - Using different optimizer
# - Adding more hidden layers
# - Change learning rate












