import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")

# One hot encode categorical data
data = pd.get_dummies(data)

pd.set_option("max_colwidth", None)
pd.set_option("display.max_columns", None)
# print(data)

features = data.loc[:, data.columns != "charges"]
labels = data.loc[:, data.columns == "charges"]

# Do not have to convert to tensors
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

tf.random.set_seed(42)
# model_1 = tf.keras.Sequential([
#     tf.keras.layers.Dense(10),
#     tf.keras.layers.Dense(1)
# ])
# model_1.compile(loss=tf.keras.losses.mae,
#                 optimizer=tf.keras.optimizers.SGD(),
#                 metrics=["mae"])
# history_1 = model_1.fit(X_train, y_train, epochs=200, verbose=0)



model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1),
])
model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["mae"])
history_2 = model_2.fit(X_train, y_train, epochs=200, verbose=1)

# pd.DataFrame(history_2.history).plot()
# plt.ylabel("loss")
# plt.xlabel("epochs")
# plt.show()