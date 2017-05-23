from __future__ import print_function

import keras
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential
import tensorflow as tf
import threading

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


class ExampleModel(object):
    def __init__(self, graph):
        with graph.as_default():
            #             with sess.as_default():
            #                 self.w = tf.Variable(tf.constant(1, shape=[1, 2]))

            self.model = Sequential()
            self.model.add(Conv2D(32, kernel_size=(3, 3),
                                  activation='relu',
                                  input_shape=input_shape))
            self.model.add(Conv2D(64, (3, 3), activation='relu'))
            self.model.add(MaxPooling2D(pool_size=(2, 2)))
            # self.model.add(Dropout(0.25))
            self.model.add(Flatten())
            self.model.add(Dense(128, activation='relu'))
            # self.model.add(Dropout(0.5))
            self.model.add(Dense(num_classes))
            self.model._make_predict_function()

            self.model.compile(loss=keras.losses.categorical_crossentropy,
                          optimizer=keras.optimizers.Adadelta(),
                          metrics=['accuracy'])

            self.m_in = tf.placeholder(tf.float32, shape=(None,) + input_shape)
            m_out = self.model(self.m_in)
            self.m_gt = tf.placeholder(tf.float32, shape=(None, num_classes))
            cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.m_gt, logits=m_out))
            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


graph = tf.get_default_graph()
_sess_config = tf.ConfigProto(
    # allow_soft_placement=True,
    # log_device_placement=True,
    # inter_op_parallelism_threads=8,
    # intra_op_parallelism_threads=8
)
_sess_config.gpu_options.allow_growth = True
# sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=_sess_config, graph=graph)

# global_network = ExampleModel(graph)
# sess.run(tf.global_variables_initializer())


def example(sess, local_network, i):
    print("worker ",i)
    with sess.as_default():
        assert sess.graph is graph, 'same '
        #         with sess.graph.as_default():
        ind=0
        while True:
            print("worker ",i,"step ",ind)
            if (ind+1)*batch_size>=x_train.shape[0]:
                break
            sess.run(local_network.train_step, feed_dict={
                local_network.m_in: x_train.copy()[ind*batch_size:(ind+1)*batch_size,...],
                local_network.m_gt: y_train.copy()[ind*batch_size:(ind+1)*batch_size,...]
            })
            ind+=1


# local_network.model.fit(x_train.copy(), y_train.copy(),
#               batch_size=batch_size,
#               epochs=epochs,
#               verbose=2,
#               validation_data=(x_test.copy(), y_test.copy()))

import  threading

threads = []
for i in range(5):
    local_network = ExampleModel(graph)
    sess.run(tf.global_variables_initializer())
    # assign_w = local_network.w.assign(global_network.w)
    t = threading.Thread(target=example, args=(sess, local_network, i))
    # example(sess, local_network, i)
    threads.append(t)

for t in threads:
    t.start()
