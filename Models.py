import numpy as np
import os
import cv2
from numpy import mean
from numpy import std
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from numpy import mean
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot
from sklearn import preprocessing
from sklearn.model_selection import KFold
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.models import Sequential
from keras.optimizers import SGD
from keras.layers import Dropout
from keras.optimizers import Adam


def define_model_vgg_16():
    model = Sequential()
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(4096, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(1000, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(7, activation='softmax'))
    # compile model
    # opt = SGD(lr=0.01, momentum=0.9)
    # opt = Adam(lr=0.001, decay=1e-6)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def define_model_basic(edge_detect, dropout, dropout_level, opt, loss):
    layers = 3
    if edge_detect:
        layers = 1
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(32, 32, edge_detect)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    if dropout:
        model.add(Dropout(dropout_level))
    model.add(Dense(7, activation='softmax'))
    # compile model
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy'])
    return model

