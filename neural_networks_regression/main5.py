import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder

data = pd.read_csv("../datasets/mnist_train.csv")

ct = make_column_transformer(
    (OneHotEncoder(handle_unknown="ignore"), ["label"])
)

X = data.drop(columns=["label"])
y = pd.DataFrame(data["label"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ct.fit(y_train)
y_train_onehot = ct.transform(y_train).toarray()
y_test_onehot = ct.transform(y_test).toarray()

model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="sigmoid"),
    tf.keras.layers.Dense(50, activation="sigmoid"),
    tf.keras.layers.Dense(10, activation="softmax")
])

model_1.compile(loss=tf.keras.losses.mse,
                optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),
                metrics=["mae"])

# Can write a **callback** to check things while training
history_1 = model_1.fit(X_train, y_train_onehot, epochs=30, verbose=1)

# Evaluate (each row of y_pred is the output of 1 example
y_pred = model_1.predict(X_test)

num_correct = 0
total_examples = len(y_pred)

for example, correct_output in zip(y_pred, np.array(y_test)):
    pred = np.argmax(np.array(example))
    # print(f"{correct_output}, {pred}")
    if pred == correct_output:
        num_correct += 1

print(f"Evaluation: {num_correct} / {total_examples}, {float(num_correct) / float(total_examples)}")
