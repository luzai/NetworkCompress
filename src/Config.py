# Dependency: Utils
from __future__ import division
from __future__ import print_function

import argparse

import keras.utils
import numpy as np
import os.path as osp
import tensorflow as tf
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.datasets import cifar10

import Utils
from Utils import root_dir


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='net work compression')
    parser.add_argument('--epoch', dest='nb_epoch', help='number epoch',
                        default=150, type=int)
    parser.add_argument("--teacher-epoch", dest="nb_teacher_epoch", help="number teacher epoch",
                        default=50, type=int)
    parser.add_argument('--verbose', dest='gl_verbose', help="global verbose",
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument("--dbg", dest="dbg",
                        help="for dbg",
                        action="store_true")
    parser.add_argument('--gpu', dest='gpu_id', help='gpu id',
                        default=0, type=int)
    _args = parser.parse_args()
    return _args


# args=parse_args()
import logging, sys

# logging.basicConfig(filename='output/net2net.log', level=logging.DEBUG)

logger = logging.getLogger('net2net')
logger.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s -  %(levelname)s -------- \n\t%(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class MyConfig(object):
    # for all model
    # Utils.mkdir_p(osp.join(root_dir,'output/tf_tmp/'))
    tf_graph = tf.get_default_graph()
    _sess_config = tf.ConfigProto(
        allow_soft_placement=True,
        # log_device_placement = True,
        # inter_op_parallelism_threads = 8,
        # intra_op_parallelism_threads = 8
    )
    _sess_config.gpu_options.allow_growth = True
    # sess_config.gpu_options.per_process_gpu_memory_fraction = 0.8
    sess = tf.Session(config=_sess_config, graph=tf_graph)
    K.set_session(sess)
    K.set_image_data_format("channels_last")
    # # for all model, but determined when init the first config
    cache_data = None

    def copy(self, name='diff_name'):
        new_config = MyConfig(self.epochs, self.verbose, self.dbg, name)
        return new_config

    def set_name(self, name):
        self.name = name
        self.tf_log_path = osp.join(root_dir, 'output/tf_tmp/', name)
        self.output_path = osp.join(root_dir, 'output/', name)
        Utils.mkdir_p(self.tf_log_path)
        Utils.mkdir_p(self.output_path)

    def __init__(self, epochs=100, verbose=1, dbg=False, name='default_name',evoluation_time=1):
        # TODO check when name = 'default_name'
        # for ga:
        self.evaluation_time=evoluation_time

        # for single model
        self.set_name(name)
        self.dbg = dbg
        self.input_shape = (3, 32, 32)
        self.nb_class = 10
        self.batch_size = 256
        self.epochs = epochs
        self.verbose = verbose
        self.lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10,
                                            min_lr=0.5e-7)
        self.early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
        self.csv_logger = CSVLogger(osp.join(root_dir, 'output', 'net2net.csv'))

        # for all model, but determined when init the first config
        if dbg:
            self.dataset = self.load_data(9999)
            logger.setLevel(logging.WARNING) # TODO modify when release
        else:
            self.dataset = self.load_data(1)
            logger.setLevel(logging.WARNING)

    def _preprocess_input(self, x, mean_image=None):
        x = x.reshape((-1,) + self.input_shape)
        x = x.astype("float32")
        if mean_image is None:
            mean_image = np.mean(x, axis=0)
        x -= mean_image
        x /= 128.
        return x, mean_image

    def _preprocess_output(self, y):
        return keras.utils.np_utils.to_categorical(y, self.nb_class)

    @staticmethod
    def _limit_data(x, div):
        div = float(div)
        return x[:int(x.shape[0] / div), ...]

    def load_data(self, limit_data):
        if MyConfig.cache_data is None:
            (train_x, train_y), (test_x, test_y) = cifar10.load_data()
            train_x, mean_img = self._preprocess_input(train_x, None)
            test_x, _ = self._preprocess_input(test_x, mean_img)

            train_y, test_y = map(self._preprocess_output, [train_y, test_y])

            res = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}

            for key, val in res.iteritems():
                res[key] = MyConfig._limit_data(val, limit_data)
            MyConfig.cache_data = res
        return MyConfig.cache_data
