# -*- coding: utf-8 -*-
'''VGG19 model for Keras.

# Reference:

- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)

'''
import sys
sys.path.insert(0,"../src")

from net2net import *
import numpy as np
import warnings
from keras.models import Model
from keras.layers import Flatten, Dense, Input
from keras.layers import Convolution2D, MaxPooling2D
from keras.preprocessing import image
from keras.utils.layer_utils import convert_all_kernels_in_model
from keras.utils.data_utils import get_file
from keras import backend as K
# from imagenet_utils import decode_predictions, preprocess_input

input_shape = (3, 32, 32)  # image shape
nb_class = 10  # number of class
lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-7)
early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
csv_logger = CSVLogger(osp.join(root_dir, 'output', 'net2net.csv'))
batch_szie = 128

img_input=Input(shape=input_shape)
# Block 1
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv1')(img_input)
x = Convolution2D(64, 3, 3, activation='relu', border_mode='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv1')(x)
x = Convolution2D(128, 3, 3, activation='relu', border_mode='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv1')(x)
x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv2')(x)
x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv3')(x)
x = Convolution2D(256, 3, 3, activation='relu', border_mode='same', name='block3_conv4')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv1')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv2')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv3')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block4_conv4')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv1')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv2')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv3')(x)
x = Convolution2D(512, 3, 3, activation='relu', border_mode='same', name='block5_conv4')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)


# Classification block
x = Flatten(name='flatten')(x)
x = Dense(2048, activation='relu', name='fc1')(x)
x = Dense(1024, activation='relu', name='fc2')(x)
x = Dense(10, activation='softmax', name='predictions')(x)

train_data, validation_data = load_data(dbg=True)

# Create model
model = Model(img_input, x)
model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.01, momentum=0.9),
                  metrics=['accuracy'])
shuffle_weights(model)
print([l.name for l in model.layers])

history = model.fit(
    *(train_data),
    nb_epoch=1,
    validation_data=validation_data,
    verbose=2,
    callbacks=[lr_reducer, early_stopper, csv_logger]
)

vgg_acc=history.history["val_acc"]
print vgg_acc
np.save("vgg.npy",vgg_acc)
plt.plot(vgg_acc)
plt.savefig("vgg.png")
