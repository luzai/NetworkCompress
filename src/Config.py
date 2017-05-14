from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import np_utils


class Config(object):
    sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        # log_device_placement = True,
        # inter_op_parallelism_threads = 8,
        # intra_op_parallelism_threads = 8
    )
    sess_config.gpu_options.allow_growth = True
    # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=sess_config)
    K.set_session(sess)
    K.set_image_data_format("channels_first")

    cache_data = None

    def __init__(self, epochs=100, verbose=1, limit_data=1):
        self.input_shape = (3, 32, 32)
        self.nb_class = 10
        self.batch_size = 256
        self.epochs = epochs
        self.verbose = verbose
        self.dataset = self.load_data(limit_data)

    def _preprocess_input(self, x, mean_image=None):
        x = x.reshape((-1,) + self.input_shape)
        x = x.astype("float32")
        if mean_image is None:
            mean_image = np.mean(x, axis=0)
        x -= mean_image
        x /= 128.
        return x, mean_image

    def _preprocess_output(self, y):
        return np_utils.to_categorical(y, self.nb_class)

    def _limit_data(self, x, div):
        return x[:int(x.shape[0] // div)]

    def load_data(self, limit_data):
        if Config.cache_data is None:
            (train_x, train_y), (test_x, test_y) = cifar10.load_data()
            train_x, mean_img = self._preprocess_input(train_x, None)
            test_x, _ = self._preprocess_input(test_x, mean_img)

            train_y, test_y = map(self._preprocess_output, [train_y, test_y])

            res = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}

            for val in res.values():
                self._limit_data(val, limit_data)
            Config.cache_data = res
        return Config.cache_data
