from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, merge
from keras.models import Model
from keras.utils import np_utils

img_channels = 3
img_rows, img_cols = 224, 224

output_classes = 10

batch_size = 32
nb_classes = 10
nb_epoch = 10
data_augmentation = True

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.repeat(7, axis=2).repeat(7, axis=3)
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

input_img = Input(shape=(img_channels, img_rows, img_cols))


X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# X_train /= 255
# X_test /= 255

# datagen = ImageDataGenerator(
# 	featurewise_center=True,
# 	featurewise_std_normalization=True,
# 	rotation_range=20,
# 	width_shift_range=0.2,
# 	height_shift_range=0.2,
# 	horizontal_flip=True)

datagen = ImageDataGenerator(
	featurewise_center=True,
	featurewise_std_normalization=True)

datagen.fit(X_train, augment=True)






# モデル定義
def inception(input_data, channels):
	c1, c2_1, c2_2, c3_1, c3_2, c4 = channels

	inc_1 = Convolution2D(c1, 1, 1, border_mode='same', activation='relu')(input_data)

	inc_2 = Convolution2D(c2_1, 1, 1, border_mode='same', activation='relu')(input_data)
	inc_2 = Convolution2D(c2_2, 3, 3, border_mode='same', activation='relu')(inc_2)

	inc_3 = Convolution2D(c3_1, 1, 1, border_mode='same', activation='relu')(input_data)
	inc_3 = Convolution2D(c3_2, 5, 5, border_mode='same', activation='relu')(inc_3)

	inc_4 = MaxPooling2D((3, 3), strides=(1, 1), border_mode='same')(input_data)
	inc_4 = Convolution2D(c4, 1, 1, border_mode='same', activation='relu')(inc_4)

	out = merge([inc_1, inc_2, inc_3, inc_4], mode='concat', concat_axis=1)
	return out


h = Convolution2D(64, 7, 7, border_mode='same', subsample=(2, 2), activation='relu')(input_img)
h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(h)
h = BatchNormalization()(h)
h = Convolution2D(64, 1, 1, border_mode='same', activation='relu')(h)
h = Convolution2D(192, 3, 3, border_mode='same', activation='relu')(h)
h = BatchNormalization()(h)
h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(h)
h = inception(h, channels=(64, 96, 128, 16, 32, 32))
h = inception(h, channels=(128, 128, 192, 32, 96, 64))
h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(h)
h = inception(h, channels=(192, 96, 208, 16, 48, 64))

l = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), border_mode='same')(h)
l = Convolution2D(128, 1, 1, border_mode='same', activation='relu')(l)
l = Flatten()(l)
l = Dense(1024, activation='relu')(l)
loss1 = Dense(nb_classes, activation='softmax')(l)

h = inception(h, channels=(160, 112, 224, 24, 64, 64))
h = inception(h, channels=(128, 128, 256, 24, 64, 64))
h = inception(h, channels=(112, 144, 288, 32, 64, 64))

l = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), border_mode='same')(h)
l = Convolution2D(128, 1, 1, border_mode='same', activation='relu')(l)
l = Flatten()(l)
l = Dense(1024, activation='relu')(l)
loss2 = Dense(nb_classes, activation='softmax')(l)

h = inception(h, channels=(256, 160, 320, 32, 128, 128))
h = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='same')(h)
h = inception(h, channels=(256, 160, 320, 32, 128, 128))
h = inception(h, channels=(384, 192, 384, 48, 128, 128))
h = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), border_mode='same')(h)
h = Flatten()(h)
h = Dropout(0.4)(h)
h = Dense(nb_classes, activation='relu')(h)
out = Dense(nb_classes, activation='softmax')(h)

model = Model(input=input_img, output=[out, loss1, loss2])

model.compile(optimizer='adam',
	loss=['categorical_crossentropy', 'categorical_crossentropy', 'categorical_crossentropy'],
	loss_weights=[1., 0.3, 0.3]
)


model.fit_generator(datagen.flow(X_train, Y_train, batch_size=batch_size, shuffle=True),
	samples_per_epoch=len(X_train), nb_epoch=nb_epoch)


# model.fit(X_train, [Y_train, Y_train, Y_train],
# 	batch_size=batch_size,
# 	nb_epoch=nb_epoch,
# 	validation_data=(X_test, [Y_test, Y_test, Y_test]),
# 	shuffle=True
# )
