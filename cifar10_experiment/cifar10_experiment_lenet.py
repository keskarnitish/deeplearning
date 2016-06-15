
# coding: utf-8

# In[2]:

import network


# In[14]:

from keras.utils.visualize_util import model_to_dot, plot
from IPython.display import SVG
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt

plt.style.use("ggplot")
import random
import pickle

lenet = network.lenet()

from keras.datasets import cifar10

batch_size = 128
nb_epoch = 50

(X_train, y_train), (X_test, y_test) = cifar10.load_data()
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255


random.seed(1405)
lenet.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
lehis = lenet.fit(X_train, Y_train,
              batch_size=batch_size,
              nb_epoch=nb_epoch,
              validation_data=(X_test, Y_test))


with open('history_lenet.dump', 'wb') as f:
    pickle.dump(lehis.history, f, -1)
