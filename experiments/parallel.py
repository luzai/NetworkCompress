from __future__ import print_function

import keras


def _slice_arrays(arrays, start=None, stop=None):
    """Slice an array or list of arrays.

    This takes an array-like, or a list of
    array-likes, and outputs:
        - arrays[start:stop] if `arrays` is an array-like
        - [x[start:stop] for x in arrays] if `arrays` is a list

    Can also work on list/array of indices: `_slice_arrays(x, indices)`

    # Arguments
        arrays: Single array or list of arrays.
        start: can be an integer index (start index)
            or a list/array of indices
        stop: integer (stop index); should be None if
            `start` was a list.

    # Returns
        A slice of the array(s).
    """
    if isinstance(arrays, list):
        if hasattr(start, '__len__'):
            # hdf5 datasets only support list objects as indices
            if hasattr(start, 'shape'):
                start = start.tolist()
            return [x[start] for x in arrays]
        else:
            return [x[start:stop] for x in arrays]
    else:
        if hasattr(start, '__len__'):
            if hasattr(start, 'shape'):
                start = start.tolist()
            return arrays[start]
        else:
            return arrays[start:stop]


class CustomKerasModel(keras.models.Model):
    def __init__(self, *args, **kwargs):
        super(CustomKerasModel, self).__init__(*args, **kwargs)
        self.cache = {}

    def prepare_fit(self, x=None,
                    y=None,
                    batch_size=32,
                    epochs=1,
                    verbose=1,
                    callbacks=None,
                    validation_split=0.,
                    validation_data=None,
                    shuffle=True,
                    class_weight=None,
                    sample_weight=None,
                    initial_epoch=0,
                    **kwargs):
        from keras.engine.training import *
        # Legacy support
        if 'nb_epoch' in kwargs:
            warnings.warn('The `nb_epoch` argument in `fit` '
                          'has been renamed `epochs`.', stacklevel=2)
            epochs = kwargs.pop('nb_epoch')
        if kwargs:
            raise TypeError('Unrecognized keyword arguments: ' + str(kwargs))

        # Validate user data.
        x, y, sample_weights = self._standardize_user_data(
            x, y,
            sample_weight=sample_weight,
            class_weight=class_weight,
            check_batch_axis=False,
            batch_size=batch_size)
        # Prepare validation data.
        if validation_data:
            do_validation = True
            if len(validation_data) == 2:
                val_x, val_y = validation_data
                val_sample_weight = None
            elif len(validation_data) == 3:
                val_x, val_y, val_sample_weight = validation_data
            else:
                raise ValueError('When passing validation_data, '
                                 'it must contain 2 (x_val, y_val) '
                                 'or 3 (x_val, y_val, val_sample_weights) '
                                 'items, however it contains %d items' %
                                 len(validation_data))

            val_x, val_y, val_sample_weights = self._standardize_user_data(
                val_x, val_y,
                sample_weight=val_sample_weight,
                check_batch_axis=False,
                batch_size=batch_size)
            self._make_test_function()
            val_f = self.test_function
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                val_ins = val_x + val_y + val_sample_weights + [0.]
            else:
                val_ins = val_x + val_y + val_sample_weights

        elif validation_split and 0. < validation_split < 1.:
            do_validation = True
            split_at = int(len(x[0]) * (1. - validation_split))
            x, val_x = (_slice_arrays(x, 0, split_at), _slice_arrays(x, split_at))
            y, val_y = (_slice_arrays(y, 0, split_at), _slice_arrays(y, split_at))
            sample_weights, val_sample_weights = (
                _slice_arrays(sample_weights, 0, split_at),
                _slice_arrays(sample_weights, split_at))
            self._make_test_function()
            val_f = self.test_function
            if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
                val_ins = val_x + val_y + val_sample_weights + [0.]
            else:
                val_ins = val_x + val_y + val_sample_weights
        else:
            do_validation = False
            val_f = None
            val_ins = None

        # Prepare input arrays and training function.
        if self.uses_learning_phase and not isinstance(K.learning_phase(), int):
            ins = x + y + sample_weights + [1.]
        else:
            ins = x + y + sample_weights
        self._make_train_function()
        f = self.train_function

        # Prepare display labels.
        out_labels = self._get_deduped_metrics_names()

        if do_validation:
            callback_metrics = copy.copy(out_labels) + ['val_' + n for n in out_labels]
        else:
            callback_metrics = copy.copy(out_labels)

        self.cache = dict(f=f, ins=ins, out_labels=out_labels,
                          batch_size=batch_size, epochs=epochs,
                          verbose=verbose, callbacks=callbacks,
                          val_f=val_f, val_ins=val_ins, shuffle=shuffle,
                          callback_metrics=callback_metrics,
                          initial_epoch=initial_epoch)

    def fit(self):
        return self._fit_loop(**self.cache)


import keras
import tensorflow as tf
from keras import backend as K
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Dropout
from keras.layers import Dense, Flatten

batch_size = 128
num_classes = 10
epochs = 10

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


def _limit_data(x, div):
    div = float(div)
    return x[:int(x.shape[0] / div), ...]


dbg = False
if dbg:
    x_train, y_train, x_test, y_test = map(lambda x: _limit_data(x, 999 // 3), [x_train, y_train, x_test, y_test])


class ExampleModel(object):
    def __init__(self, graph, name='default'):
        with graph.as_default():
            #             with sess.as_default():
            #                 self.w = tf.Variable(tf.constant(1, shape=[1, 2]))


            keras_in = keras.layers.Input(shape=input_shape)
            x = keras_in
            x = Conv2D(32, kernel_size=(3, 3),
                       activation='relu',
                       input_shape=input_shape)(x)
            x = Conv2D(64, (3, 3), activation='relu')(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)
            x = Dropout(0.25)(x)
            x = Flatten()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(num_classes)(x)
            self.model = CustomKerasModel(inputs=keras_in, outputs=x)
            self.model._make_predict_function()
            # self.model._make_test_function()
            self.model.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=keras.optimizers.Adadelta(),
                               metrics=['accuracy'])
            # self.model._make_train_function()
            self.model.prepare_fit(x_train, y_train,
                                   batch_size=batch_size,
                                   epochs=epochs,
                                   verbose=2,
                                   validation_data=(x_test, y_test),
                                   callbacks=[keras.callbacks.TensorBoard(log_dir='tmp_tf/' + name + '/')])
            self.m_in = tf.placeholder(tf.float32, shape=(None,) + input_shape)
            m_out = self.model(self.m_in)
            self.m_gt = tf.placeholder(tf.float32, shape=(None, num_classes))
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.m_gt, logits=m_out))
            self.train_step = tf.train.GradientDescentOptimizer(0.5).minimize(self.cross_entropy)


graph = tf.get_default_graph()
_sess_config = tf.ConfigProto(
    allow_soft_placement=True,
    # log_device_placement=True,
    # inter_op_parallelism_threads=8,
    # intra_op_parallelism_threads=8
)
_sess_config.gpu_options.allow_growth = True
# sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=_sess_config, graph=graph)


def example(sess, local_network, woker_id):
    print("worker ", woker_id)
    with sess.as_default():
        assert sess.graph is graph, 'same '
        local_network.model.fit()
        # for epoch in range(epochs):
        #     ind = 0
        #     while True:
        #         # print("worker ", i, "step ", ind)
        #         if (ind + 1) * batch_size >= x_train.shape[0]:
        #
        #             print('loss',loss)
        #             break
        #         _,loss=sess.run([local_network.train_step,local_network.cross_entropy], feed_dict={
        #             local_network.m_in: x_train[ind * batch_size:(ind + 1) * batch_size, ...],
        #             local_network.m_gt: y_train[ind * batch_size:(ind + 1) * batch_size, ...]
        #         })
        #
        #         ind += 1
        #     print("woker ", woker_id, "epoch", epoch)


import time, threading, subprocess
import multiprocessing as mp

subprocess.call('rm -rf tmp_tf'.split())
# parallel = True
res = []
for parallel in [True]:
    # for num in [1, 3, 5, 10, 30]:
    for num in [1, 2, 5, 10]:
        threads = []

        for i in range(num):
            local_network = ExampleModel(graph, name='time{}_woker_{}'.format(num, i))
            # sess = tf.Session(config=_sess_config, graph=graph)
            sess.run(tf.global_variables_initializer())

            if parallel:

                t = threading.Thread(target=example, args=(sess, local_network, i))
                threads.append(t)
                # t.start()
            else:
                example(sess, local_network, i)

        tic = time.time()
        if parallel:
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        toc = time.time() - tic
        res.append(toc)
        print("consume ", toc)

print(res)
