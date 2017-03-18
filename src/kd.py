from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

def kd_loss(y_true, y_pred):
	return K.categorical_crossentropy(y_pred, y_true, from_logits = True)

# 1.prepare MNIST data
nb_classes = 10
img_rows, img_cols = 28, 28
pool_size = (2, 2)
kernel_size = (3, 3)
input_shape = (1, img_rows, img_cols)
batch_size = 128
nb_epoch = 1

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)
print('X_test shape:', X_test.shape)
print('Y_test shape:', Y_test.shape)


#2. train the teacher model
x_input = Input(shape=input_shape)
t_conv1 = Convolution2D(128, kernel_size[0], kernel_size[1], 
		border_mode='valid', activation='relu')(x_input)
t_conv2 = Convolution2D(128, kernel_size[0], kernel_size[1], 
		border_mode='valid', activation='relu')(t_conv1)
t_pool2 = MaxPooling2D(pool_size=pool_size)(t_conv2)
t_flatten = Flatten()(t_pool2)
t_dense3 = Dense(128, activation='relu')(t_flatten)
t_dense4 = Dense(nb_classes)(t_dense3)
t_output = Activation('softmax')(t_dense4)

model = Model(input=x_input, output=t_output)

model.compile(optimizer = 'adadelta', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.fit(X_train, Y_train,
              batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1,
              validation_data=(X_test, Y_test))

# save teacher model and teacher logits & labels
model.save('teacher_model')
teacher_logits_output = K.function([model.layers[0].input],
                                  [model.layers[6].output])

for i in range(X_train.shape[0] / 10000):
	layer_output = teacher_logits_output([X_train[i * 10000 : (i + 1) * 10000]])[0]
	np.save('teacher_logits_train_' + str(i) + '.npy', layer_output)

layer_output = teacher_logits_output([X_test])[0]
np.save('teacher_logits_test.npy', layer_output)

#here we need transfer data
teacher_label_train = model.predict([X_train])
teacher_label_test = model.predict([X_test])
np.save('teacher_label_train.npy', teacher_label_train)
np.save('teacher_label_test.npy', teacher_label_test)

# 3. train the student model using knowledge distillation
x_input = Input(shape=input_shape)
s_conv1 = Convolution2D(32, kernel_size[0], kernel_size[1], 
		border_mode='valid', activation='relu')(x_input)
s_conv2 = Convolution2D(32, kernel_size[0], kernel_size[1], 
		border_mode='valid', activation='relu')(s_conv1)
s_pool2 = MaxPooling2D(pool_size=pool_size)(s_conv2)
s_flatten = Flatten()(s_pool2)
s_dense3 = Dense(64, activation = 'relu')(s_flatten)
s_logits = Dense(nb_classes)(s_dense3)
s_output = Activation('softmax')(s_logits)

s_model = Model(input=x_input, output=[s_logits, s_output])

s_model.compile(optimizer = 'adadelta', loss = [kd_loss, 'categorical_crossentropy'],\
		 loss_weights = [1, 0], metrics = ['accuracy'])
#read teacher logits
t_logits_train = []
for i in range(6):
	t_logits_train.append(np.load('teacher_logits_train_' + str(i) + '.npy'))
t_logits_train = np.asarray(t_logits_train)
t_logits_train = t_logits_train.reshape(t_logits_train.shape[0] * t_logits_train.shape[1], t_logits_train.shape[2])
t_logits_test = np.asarray(np.load('teacher_logits_test.npy'))



print('t_logits_train shape:', t_logits_train.shape)
print('t_logits_test shape:', t_logits_test.shape)

s_model.fit(X_train, [t_logits_train, teacher_label_train],
              batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1,
              validation_data=(X_test, [t_logits_test, teacher_label_test]))

