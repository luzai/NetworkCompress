from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import cifar10
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from load_transfer_data import get_transfer_data

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

print('transfer_x shape: ', transfer_x.shape)
print('transfer_y shape: ', transfer_y.shape)
print('test_x shape: ', test_x.shape)
print('test_y shape: ', test_y.shape)

#2. read teacher_model's logits
t_logits_transfer = np.asarray(np.load('../output/resnet18_logits_transfer.npy'))
t_logits_test = np.asarray(np.load('../output/resnet18_logits_test.npy'))
print ('t_logits_transfer.shape: ', t_logits_transfer.shape)
print ('t_logits_test.shape: ', t_logits_test.shape)

#3. define student model
nb_classes = 10
pool_size = (2, 2)
kernel_size = (3, 3)
input_shape = (3, 32, 32)
batch_size = 128
nb_epoch = 10

x_input = Input(shape = input_shape)
s_conv1 = Convolution2D(64, kernel_size[0], kernel_size[1], 
        border_mode='valid', activation='relu')(x_input)
s_conv2 = Convolution2D(128, kernel_size[0], kernel_size[1], 
        border_mode='valid', activation='relu')(s_conv1)
s_pool2 = MaxPooling2D(pool_size=pool_size)(s_conv2)
s_flatten = Flatten()(s_pool2)
s_dense3 = Dense(256, activation = 'relu')(s_flatten)
s_logits = Dense(nb_classes)(s_dense3)
s_output = Activation('softmax')(s_logits)
s_model = Model(input=x_input, output=[s_logits, s_output])
s_model.compile(optimizer = 'adadelta', loss = ['mean_squared_error', 'categorical_crossentropy'],\
         loss_weights = [1, 0], metrics = ['accuracy'])

s_model.fit(transfer_x, [t_logits_transfer, transfer_y],
              batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1,
              validation_data=(test_x, [t_logits_test, test_y]))

s_model.save('student_1.model')
