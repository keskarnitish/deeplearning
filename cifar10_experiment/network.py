
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.models import Model
from keras.utils import np_utils

img_size = (3, 32, 32)
nb_class = 10

def lenet():
	global img_size, nb_class

	model = Sequential()
	model.add(Convolution2D(4, 3, 3, border_mode='same', input_shape=img_size, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	model.add(Convolution2D(6, 3, 3, border_mode='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
	model.add(Flatten())
	model.add(Dense(100, activation='relu'))
	model.add(Dense(nb_class, activation='softmax'))

	return model

def alexnet():
	global img_size, nb_class

	model = Sequential()
	model.add(Convolution2D(96, 11, 11, border_mode='same', input_shape=img_size))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='same'))
	model.add(Convolution2D(384, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution2D(384, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='same'))
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.25))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_class, activation='softmax'))

	return model


def vggnet():
	global img_size, nb_class

	model = Sequential()
	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu', input_shape=img_size))
	model.add(Convolution2D(64, 3, 3, border_mode='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same'))
	model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution2D(128, 3, 3, border_mode='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same'))
	model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution2D(256, 3, 3, border_mode='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same'))
	model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu'))
	model.add(Convolution2D(512, 3, 3, border_mode='same', activation='relu'))
	model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), border_mode='same'))
	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(nb_class, activation='softmax'))

	return model

def resnet():
	global img_size, nb_class

	def res(channel, x):
		h = Convolution2D(channel, 3, 3, border_mode='same', activation='relu')(x)
		h = Convolution2D(channel, 3, 3, border_mode='same')(h)
		h = merge([h, x], mode='sum')
		return Activation('relu')(h)

	def res_down(out_channel, x):
		h = Convolution2D(out_channel, 3, 3, border_mode='same', activation='relu', subsample=(2,2))(x)
		h = Convolution2D(out_channel, 3, 3, border_mode='same')(h)

		l = Convolution2D(out_channel, 1, 1, border_mode='same', subsample=(2,2))(x)
		h = merge([h, l], mode='sum')
		return Activation('relu')(h)

	input_image = Input(shape=img_size)
	h = Convolution2D(64, 7, 7, subsample=(2,2), border_mode='same')(input_image)
	h = MaxPooling2D(pool_size=(3,3), strides=(2,2), border_mode='same')(h)
	h = res(64, h)
	h = res(64, h)
	h = res(64, h)
	h = res_down(128, h)
	h = res(128, h)
	h = res(128, h)
	h = res(128, h)
	h = res_down(256, h)
	h = res(256, h)
	h = res(256, h)
	h = res(256, h)
	h = res(256, h)
	h = res(256, h)
	h = res_down(512, h)
	h = res(512, h)
	h = res(512, h)
	h = Flatten()(h)
	out = Dense(nb_class, activation='softmax')(h)

	return Model(input=input_image, output=out)


if __name__ == '__main__':
	le = lenet()
	alex = alexnet()
	vgg = vggnet()
	res = resnet()
