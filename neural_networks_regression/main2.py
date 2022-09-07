import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.random.set_seed(42)

X = tf.range(-100, 100, 4)
X_train = tf.transpose(X[0:40])
X_test = tf.transpose(X[40:])

y = X + 10
y_train = y[0:40]
y_test = y[40:]

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10,input_shape=[1], name="hidden_layer_1"),
    tf.keras.layers.Dense(1, name="output_layer")
], name="model_1")

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              metrics=["mae"])

# # Create a diagram of the neural network
# model.summary()
# tf.keras.utils.plot_model(model=model, to_file="model_plot.png", show_shapes=True)

model.fit(X_train, y_train, epochs=150, verbose=1)
y_pred = model.predict(X_test)

# # Plot training, test, and prediction data
# plt.figure(figsize=(10, 7))
# #plt.scatter(X_train, y_train, c='b', label="Training data")
# plt.scatter(X_test, y_test, c='g', label="Testing data")
# plt.scatter(X_test, y_pred, c='r', label="Predicted")
# plt.show()

# Calculate the MAE and MSE of the predictions
mae = tf.metrics.mean_absolute_error(y_true=y_test,
                                     y_pred=tf.squeeze(tf.constant(y_pred)))
print(mae)

mse = tf.metrics.mean_squared_error(y_true=y_test,
                                    y_pred=tf.squeeze(tf.constant(y_pred)))
print(mse)