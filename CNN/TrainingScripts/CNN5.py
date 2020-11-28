import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense, Input, Flatten, Conv2D, BatchNormalization, \
                                    Activation, Dropout, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from IPython.display import YouTubeVideo
from PIL import Image
import pickle, gzip
import numpy as np

X, y = pickle.load(gzip.open('flatland_train.data', 'rb'))
y[y != 0] -= 2
y = np.concatenate((y, y, y, y))
X /= 255.
X = np.concatenate((X, [np.transpose(x) for x in X], np.flip(X, 1), [np.transpose(x) for x in np.flip(X, 1)]))
X = X[..., np.newaxis]

model = Sequential()
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu', input_shape=[50, 50, 1]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(8, kernel_size=(3, 3), activation='sigmoid'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(5, activation='softmax'))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.Adam(),
              metrics=["accuracy"])
model.summary()

root_logdir = os.path.join(os.curdir, "logs")

def get_run_logdir():
    run_id = 'CNN5'
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = TensorBoard(run_logdir)

model.fit(X, y, epochs=30, batch_size=32, validation_split=0.2, callbacks=[tensorboard_cb])

model.save('flatlandCNN5.h5')