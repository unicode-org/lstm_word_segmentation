# import PyICU as pyicu
import numpy as np
import icu
import datetime
from icu import UnicodeString, BreakIterator, Locale
import matplotlib.pyplot as plt
import random as rd
import tensorflow as tf
import string

import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.layers import Embedding
from keras.layers import Activation

from datetime import datetime
from packaging import version
import tensorboard

def get_sequence(n):
    # create a sequence of random numbers in [0,1]
    X = np.array([rd.random() for _ in range(n)])
    # calculate cut-off value to change class values
    limit = n / 4.0
    # determine the class outcome for each item in cumulative sequence
    y = np.array([0 if x < limit else 1 for x in np.cumsum(X)])
    # reshape input and output data to be suitable for LSTMs
    X = X.reshape(1, n, 1)
    y = y.reshape(1, n, 1)
    return X, y


def get_sequence_multi_output(times, n):
    X = np.zeros(shape=[times, 1])
    Y = np.zeros(shape=[times, 3])
    for i in range(times//n):
        temp = np.array([rd.random() for _ in range(n)])
        temp = temp.reshape(temp.shape[0], 1)
        X[i*n: (i+1)*n] = temp
        limit1 = n / 6.0
        limit2 = n / 3
        Y[i*n: (i+1)*n, 0] = np.array([1 if x < limit1 else 0 for x in np.cumsum(temp)])
        Y[i*n: (i+1)*n, 1] = np.array([1 if limit1 <= x < limit2 else 0 for x in np.cumsum(temp)])
        Y[i*n: (i+1)*n, 2] = np.array([1 if limit2 <= x else 0 for x in np.cumsum(temp)])
    return X, Y


def get_sequence_2inp_1out(n):
    X = np.zeros(shape=[1, n, 2])
    temp = np.cumsum(np.random.rand(1, n)/n)
    # temp = np.random.rand(1, n)
    X[0, :, 0] = temp
    temp = np.cumsum(np.random.rand(1, n) / n)
    # temp = np.random.rand(1, n)
    X[0, :, 1] = temp
    Y = np.zeros(shape=[1, n, 1])
    for i in range(n):
        if X[0, i, 0] + X[0, i, 1] > 0.5:
            Y[0, i, 0] = 1
    # print(X)
    # print(Y)
    return X, Y
    # x = input()

def get_random_text(times, n):
    X = np.zeros(shape=[times, 1])
    Y = np.zeros(shape=[times, 2])
    for i in range(times):
        new_char = rd.choice(string.ascii_letters)
        new_int = 0
        if ord('A') <= ord(new_char) <= ord('Z'):
            new_int = ord(new_char) - ord('A')
        elif ord('a') <= ord(new_char) <= ord('z'):
            new_int = ord(new_char) - ord('a') + 26
        X[i, 0] = new_int

        if new_char in ['a', 'e', 'o', 'i', 'u', 'A', 'E', 'O', 'I', 'U']:
            Y[i, 0] = 1
            Y[i, 1] = 0
        else:
            Y[i, 0] = 0
            Y[i, 1] = 1
    return X, Y

class KerasBatchGenerator(object):
    def __init__(self, x_data, y_data, time_steps, batch_size, dim_features, dim_output, times):
        self.x_data = x_data  # dim = times * dim_features
        self.y_data = y_data  # dim = times * dim_output
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.dim_features = dim_features
        self.dim_output = dim_output
        self.times = times

    def generate(self):
        x = np.zeros((self.batch_size, self.time_steps, self.dim_features))
        y = np.zeros((self.batch_size, self.time_steps, self.dim_output))
        while True:
            for i in range(self.batch_size):
                x[i, :, :] = self.x_data[self.time_steps * i: self.time_steps * (i + 1), :]
                y[i, :, :] = self.y_data[self.time_steps * i: self.time_steps * (i + 1), :]
            yield x, y

class KerasBatchGenerator2(object):
    def __init__(self, x_data, y_data, time_steps, batch_size, dim_features, dim_output, times):
        self.x_data = x_data  # dim = times * dim_features
        self.y_data = y_data  # dim = times * dim_output
        self.time_steps = time_steps
        self.batch_size = batch_size
        self.dim_features = dim_features
        self.dim_output = dim_output
        self.times = times

    def generate(self):
        x = np.zeros((self.batch_size, self.time_steps))
        # x = np.zeros((self.batch_size, self.time_steps, self.dim_features))
        y = np.zeros((self.batch_size, self.time_steps, self.dim_output))
        while True:
            for i in range(self.batch_size):
                # print(self.x_data.shape)
                # x = input()
                x[i, :] = self.x_data[self.time_steps * i: self.time_steps * (i + 1), 0]
                # x[i, :, :] = self.x_data[self.time_steps * i: self.time_steps * (i + 1), :]
                y[i, :, :] = self.y_data[self.time_steps * i: self.time_steps * (i + 1), :]
            yield x, y


# A very simple case: input with one feature, output with one feature
'''
n = 10
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n, 1)))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

for epoch in range(100):
    # generate new random sequence
    X, y = get_sequence(n)
    # fit model for one epoch on this sequence
    model.fit(X, y, epochs=1, batch_size=1, verbose=2)

X, y = get_sequence(n)
y_hat = model.predict_classes(X, verbose=0)
for i in range(n):
    print('Expected:', y[0, i], 'Predicted', y_hat[0, i])
'''

# With embedding
'''
hidden_size = 20
embedding_size = 4
model = Sequential()
model.add(Embedding(vocabulary, hidden_size, input_length=n))
model.add(Bidirectional(LSTM(hidden_size, return_sequences=True)))
model.add(TimeDistributed(Dense(1)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
'''

# Input with one feature, output with two feature
'''
# Generating training and validation data
n = 10
times = 1000
x_data, y_data = get_sequence_multi_output(times, n)
train_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=3,
                                      times=times)
n = 10
times = 1000
x_data, y_data = get_sequence_multi_output(times, n)
valid_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=3,
                                      times=times)

# Building the model
model = Sequential()
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n, 1)))
model.add(TimeDistributed(Dense(3, activation='softmax')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])

# logdir="logs/fit/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

model.fit(train_generator.generate(), steps_per_epoch=train_generator.times//train_generator.batch_size,
                    epochs=10, validation_data=valid_generator.generate(), validation_steps=valid_generator.times//
                                                                                            valid_generator.batch_size)
test_cycles = 2
n = 10
times = test_cycles*n
x_data, y_data = get_sequence_multi_output(times, n)
test_generator = KerasBatchGenerator(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=3,
                                      times=times)
test_input, actual_y = next(test_generator.generate())
y_hat = model.predict(test_input)
for i in range(test_cycles):
    print("test case = {}".format(i))
    # print(actual_y[i, :, :])
    # print(y_hat[i, :, :])
    print(" Expected = {}".format(np.argmax(actual_y[i, :, :], axis=1)))
    print("Estiamted = {}".format(np.argmax(y_hat[i, :, :], axis=1)))
'''

# Language based task
# '''
n = 50
times = 1000
x_data, y_data = get_random_text(times, n)
train_generator = KerasBatchGenerator2(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=2,
                                      times=times)
n = 50
times = 1000
x_data, y_data = get_random_text(times, n)
valid_generator = KerasBatchGenerator2(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=2,
                                      times=times)
# Building the model
model = Sequential()
model.add(Embedding(52, 20, input_length=n))
model.add(Bidirectional(LSTM(20, return_sequences=True), input_shape=(n, 1)))
model.add(TimeDistributed(Dense(2, activation='softmax')))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_generator.generate(), steps_per_epoch=train_generator.times//train_generator.batch_size,
                    epochs=10, validation_data=valid_generator.generate(), validation_steps=valid_generator.times//
                                                                                            valid_generator.batch_size)

test_cycles = 2
n = 50
times = test_cycles*n
x_data, y_data = get_random_text(times, n)
test_generator = KerasBatchGenerator2(x_data, y_data, time_steps=n, batch_size=times//n, dim_features=1, dim_output=2,
                                      times=times)
test_input, actual_y = next(test_generator.generate())
y_hat = model.predict(test_input)
for i in range(test_cycles):
    print("test case = {}".format(i))
    # print(actual_y[i, :, :])
    # print(y_hat[i, :, :])
    # print(actual_y)
    # print(y_hat)
    print(" Expected = {}".format(np.argmax(actual_y[i, :, :], axis=1)))
    print("Estiamted = {}".format(np.argmax(y_hat[i, :, :], axis=1)))
# '''
