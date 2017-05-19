from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
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

# 2. load teacher model
teacher_model_path = '../model/resnet18_cifar10.model'
teacher_model = load_model(teacher_model_path)
layer_names = [l.name for l in teacher_model.layers]
print(layer_names)

# 3. test teacher model's acc in test set
#pred = teacher_model.predict(test_x, batch_size = 256)

# 3. save teacher model's logits
teacher_logits_output = K.function([teacher_model.layers[0].input, K.learning_phase()],
                                  [teacher_model.get_layer(layer_names[-2]).output])

#layer_output = teacher_logits_output([transfer_x, 0])[0]
layer_output = []

for i in range(transfer_x.shape[0] / 200 + 1):
    print (i)
    if i < transfer_x.shape[0] / 200:
        layer_output.extend(teacher_logits_output([transfer_x[i * 200 : (i + 1) * 200], 0])[0])
    else:
        layer_output.extend(teacher_logits_output([transfer_x[i * 200 : ], 0])[0])

np.save('../output/resnet18_logits_transfer.npy', layer_output)

'''
layer_output = []
for i in range(test_x.shape[0] / 1000):
    print (i)
    layer_output.extend(teacher_logits_output([test_x[i * 1000 : (i + 1) * 1000], 0])[0])
print ('logits_test shape: ', layer_output.shape)
np.save('../output/resnet18_logits_test.npy', layer_output)
'''
