import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import make_column_transformer
from sklearn.metrics import confusion_matrix
import matplotlib.cm

n_samples = 1000

X, y = make_circles(n_samples,
                    noise=0.03,
                    random_state=42)

circles = pd.DataFrame({"X0":X[:, 0], "X1":X[:, 1], "label":y})
# # Plot the circles made
# plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
# plt.show()

X = pd.DataFrame({"X0":X[:, 0], "X1":X[:, 1]})
y = pd.DataFrame({"label":y})

ct = make_column_transformer(
    (MinMaxScaler(), ["X0", "X1"])
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ct.fit(X_train)

X_train_normal = ct.transform(X_train)
X_test_normal = ct.transform(X_test)

model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model_1.compile(loss=tf.keras.losses.BinaryCrossentropy(),
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
                metrics=["accuracy", "mae"])

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch / 20))

history_1 = model_1.fit(X_train_normal, y_train,
                        epochs=40,
                        # callbacks=[lr_scheduler],
                        verbose=0)

# # Plot history of model vs different learning rates using learningratescheduler
# pd.DataFrame(history_1.history).plot(figsize=(10, 7), xlabel="epochs")

# # Plot the learning rate vs loss of model
# lrs = 1e-4 * (10 ** (tf.range(100) / 20))
# plt.figure(figsize=(10, 7))
# plt.semilogx(lrs, history_1.history["loss"])
# plt.xlabel("Learning Rate")
# plt.ylabel("Loss")
# plt.title("Learning Rate vs Loss")
# plt.show()

# Evaluate (each row of y_pred is the output of 1 example
y_pred = model_1.predict(X_test_normal)
metrics = model_1.evaluate(X_test_normal, y_test)

# Make a confusion matrix (shows rates of true pos/neg, false pog/neg
confusion_matrix = confusion_matrix(y_test, tf.round(y_pred))
print(confusion_matrix)


def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary created by a model predicting on X.
    This function has been adapted from two phenomenal resources:
     1. CS231n - https://cs231n.github.io/neural-networks-case-study/
     2. Made with ML basics - https://github.com/GokuMohandas/MadeWithML/blob/main/notebooks/08_Neural_Networks.ipynb
    """
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Create X values (we're going to predict on all of these)
    x_in = np.c_[xx.ravel(), yy.ravel()]  # stack 2D arrays together: https://numpy.org/devdocs/reference/generated/numpy.c_.html

    # Make predictions using the trained model
    y_pred = model.predict(x_in)

    # Check for multi-class
    if model.output_shape[
        -1] > 1:  # checks the final dimension of the model's output shape, if this is > (greater than) 1, it's multi-class
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classifcation...")
        y_pred = np.round(np.max(y_pred, axis=1)).reshape(xx.shape)

    # Plot decision boundary
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# # Plot decision boundaries for train and test set
# plt.figure(figsize=(10, 7))
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plot_decision_boundary(model_1, X_train_normal, np.array(y_train))
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plot_decision_boundary(model_1, X_test_normal, np.array(y_test))
# plt.show()