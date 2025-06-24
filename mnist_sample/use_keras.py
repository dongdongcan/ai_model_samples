import numpy as np
import pandas as pd
from keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(x_train, y_train_lable), (x_test, y_test_lable) = mnist.load_data()

X_train = x_train.reshape(60000, 28, 28, 1)
X_test = x_test.reshape(10000, 28, 28, 1)
y_train = to_categorical(y_train_lable, 10)
y_test = to_categorical(y_test_lable, 10)

from keras import models
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = models.Sequential()
model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation="softmax"))


model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, validation_split=0.3, epochs=5, batch_size=128)


score = model.evaluate(X_test, y_test)
print("Test Accuracy:", score[1])


pred = model.predict(X_test[0].reshape(1, 28, 28, 1))
print(pred[0], "transform to:", pred.argmax())
# import matplotlib.pyplot as plt
# plt.imshow(X_test[0].reshape(28, 28),cmap='Greys')
