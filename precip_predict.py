from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.models import Model
from keras.utils import np_utils



input_shape = (input_channel, input_row, input_col) = (2, 64, 64)

model = Sequential()
model.add(Convolution2D(64, 7, 7, border_mode='same', input_shape=input_shape)
model.add(Activation('relu'))
model.add(Convolution2D(256, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
model.add(Convolution2D(512, 5, 5, border_mode='same')
model.add(Activation('relu'))
model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))
model.add(Convolution2D(64, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same'))

model.add(Flatten())
model.add(Dropout(p=0.25))
model.add(Dense(128, activation='relu'))
model.add(Dropout(p=0.1))
model.add(Dense(1, activation='relu'))

model.compile(loss='mse', optimizer='adam', metrics=['error'])
