from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from load_transfer_data import get_transfer_data
from keras.models import load_model

def kd_loss(y_true, y_pred):
    return K.categorical_crossentropy(y_pred, y_true, from_logits = True)

#1. get transfer data and test data
transfer_data_path = '../data/transfer_data/'
nb_classes = 10
(train_x, train_y), (test_x, test_y) = cifar10.load_data()
transfer_x, transfer_y = get_transfer_data(transfer_data_path)
transfer_y=transfer_y.reshape((-1,1))

transfer_x=np.concatenate((train_x,transfer_x))
transfer_y=np.concatenate((train_y,transfer_y))

# Convert class vectors to binary class matrices.
transfer_y = np_utils.to_categorical(transfer_y, nb_classes)
test_y = np_utils.to_categorical(test_y, nb_classes)

transfer_x = transfer_x.astype('float32')
test_x = test_x.astype('float32')

# subtract mean and normalize
mean_image = np.mean(transfer_x, axis=0)
transfer_x -= mean_image
test_x -= mean_image
transfer_x /= 128.
test_x /= 128.


# 2. load teacher model
teacher_model_path = '../model/resnet18_cifar10.h5'
teacher_model = load_model(teacher_model_path)
layer_name = [l.name for l in teacher_model.layers]
print(layer_name)

# 3. save teacher model's logits and labels 
pred = teacher_model.predict(test_x, batch_size = 128)
print(pred[0],  test_y[0])
