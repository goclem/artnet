#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Description: Convolutional network example on the MINST dataset
Author: Clement Gorin
Contact: gorin@gate.cnrs.fr
Date: September 2020
"""
 
"""
GPU setup: https://towardsdatascience.com/gpu-accelerated-machine-learning-on-macos-48d53ef1b545
"""
 
#%% Modules

import os
os.environ["KERAS_BACKEND"] = "plaidml.keras.backend" # GPU setting

import keras
import matplotlib.pyplot as plt
import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from random import sample 

#%% Data

# Loading
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshaping
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test  = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Normalising    
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train /= 255
x_test  /= 255

# Formatting
y_train = keras.utils.to_categorical(y_train, 10)
y_test  = keras.utils.to_categorical(y_test, 10)

#%% Model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))
model.compile(
    loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'])

model.summary()

#%% Training & Testing

# Training
model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))

# Testing
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])

stats = model.history.history
stats.keys()

# summarize history for accuracy
plt.plot(stats['loss'])
plt.plot(stats['val_loss'])
plt.title('Model loss')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for accuracy
plt.plot(stats['acc'])
plt.plot(stats['val_acc'])
plt.title('Model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

#%% Predicting

yh_test = model.predict(x_test)
yh_test = np.argmax(yh_test, axis=1)

cols, rows = (10, 10)
fig, axes  = plt.subplots(cols, rows, figsize=(2 * cols, 2 * rows))
axes = axes.flatten()
smps = sample(range(len(yh_test)), (cols * rows))
for i in range(len(smps)):
    idx = smps[i]
    axes[i].imshow(x_test[idx,:,:,0], cmap='gray')
    axes[i].set_title('Label: {}'.format(yh_test[idx]))
    axes[i].axis('off')