# !/usr/bin/env python
from __future__ import division
from __future__ import print_function
from init import *

from mLog import logger
from mModel import MyModel

import keras
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout, Activation, BatchNormalization, Embedding
from keras.models import Sequential, model_from_json, Model
from keras.optimizers import SGD
from keras.utils import np_utils, vis_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import tensorflow as tf
import scipy, scipy.ndimage
import numpy as np

from load_transfer_data import get_transfer_data


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

    def __init__(self, epochs=1, verbose=0, limit_data=1):
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
        return x[:x.shape[0] / div]

    def load_data(self, limit_data):
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()
        train_x, mean_img = self._preprocess_input(train_x, None)
        test_x, _ = self._preprocess_input(test_x, mean_img)

        train_y, test_y = map(self._preprocess_output, [train_y, test_y])

        res = {'train_x': train_x, 'train_y': train_y, 'test_x': test_x, 'test_y': test_y}

        for val in res.values():
            self._limit_data(val, limit_data)

        return res


class Net2Net(object):
    @staticmethod
    def _reorder_list(model_l):
        len_of_list = len(model_l)
        names = [layer[-1] for layer in model_l]
        name2ind = {name: ind for ind, name in enumerate(names)}

        for ind, layer in enumerate(model_l):
            if layer[-1] == 'new':
                break
        new_layer_ind = ind
        type = layer[0]
        start_ind = 1
        for ind, layer in enumerate(model_l):
            now_type = layer[0]
            if now_type == type:
                layer[-1] = type + str(start_ind)
                start_ind += 1
        return model_l, model_l[new_layer_ind][-1]

    def wider_conv2d(self, model, name, width_ratio):
        new_model_l = copy.deepcopy(model.model_l)
        for layer in new_model_l:
            if layer[-1] == name:
                new_width = layer[1] * width_ratio
                layer[1] = new_width
        assert new_width in locals()
        debug = logger.debug(new_model_l)
        new_model = MyModel(model.config, new_model_l)

        w_conv1, b_conv1 = model.get_layer(name).get_weights()
        w_conv2, b_conv2 = model.get_layer(name, next_layer=True).get_weights()

        new_w_conv1, new_b_conv1, new_w_conv2 = self._wider_conv2d_weight(
            w_conv1, b_conv1, w_conv2, new_width, "net2wider")

        new_model.get_layer(name).set_weights([new_w_conv1, new_b_conv1])
        new_model.get_layer(name, next_layer=True).set_weights([new_w_conv2, b_conv2])

        return new_model

    def deeper_conv2d(self, model, name, kernel_size=3, filters='same'):
        new_model_l = copy.deepcopy(model.model_l)
        for ind, layer in enumerate(new_model_l):
            if layer[-1] == name:
                new_layer = copy.deepcopy(layer)
                new_layer[1] = layer[1] if filters == 'same' else filters
                new_layer[2] = kernel_size
                new_layer[-1] = 'new'
                break
        assert new_layer in locals() and ind in locals()
        new_model_l.insert(ind, new_layer)
        new_model_l, new_layer_name = self._reorder_list(new_model_l)

        logger.debug(new_model_l)

        new_model = MyModel(model.config, new_model_l)

        w_conv1, b_conv1 = model.get_layer(name, last_layer=True).get_weights()
        w_conv2, b_conv2 = model.get_layer(name).get_weights()

        new_w_conv2, new_b_conv2 = self._deeper_conv2d_weight(
            w_conv1, b_conv1, w_conv2, b_conv2, "net2deeper")

        model.get_layer(name).set_weights([new_w_conv2, new_b_conv2])

    @staticmethod
    def _deeper_conv2d_weight(teacher_w1, teacher_b1, teacher_w2, teacher_b2, init='net2deeper'):
        student_w, student_b \
            = map(Net2Net._common_conv2d_weight,
                  (teacher_w1, teacher_w2.shape),
                  (teacher_b1, teacher_b2.shape))
        return student_w, student_b

    @staticmethod
    def _common_conv2d_weight(w, b, nw_size, nb_size):
        def convert_weight(w, nw_size):
            w_size = w.shape
            assert len(w_size) == len(nw_size)
            w_ratio = [nw / w for nw, w in zip(nw_size, w_size)]
            nw = scipy.ndimage.zoom(w, w_ratio)
            return nw

        return convert_weight(w, nw_size), convert_weight(b, nb_size)

    @staticmethod
    def _wider_conv2d_weight(teacher_w1, teacher_b1, teacher_w2, new_width, init, help=None):
        assert teacher_w1.shape[3] == teacher_w2.shape[2], (
            'successive layers from teacher model should have compatible shapes')
        assert teacher_w1.shape[3] == teacher_b1.shape[0], (
            'weight and bias from same layer should have compatible shapes')
        assert new_width > teacher_w1.shape[3], (
            'new width (filters) should be bigger than the existing one')

        n = new_width - teacher_w1.shape[3]
        if init == 'random-pad':
            new_w1 = np.random.normal(0, 0.1, size=teacher_w1.shape[:-1] + (n,))
            new_b1 = np.ones(n) * 0.1
            new_w2 = np.random.normal(0, 0.1, size=teacher_w2.shape[:2] + (
                n, teacher_w2.shape[3]))
        elif init == 'net2wider':
            index = np.random.randint(teacher_w1.shape[3], size=n)
            factors = np.bincount(index)[index] + 1.
            new_w1 = teacher_w1[:, :, :, index]
            new_b1 = teacher_b1[index]
            new_w2 = teacher_w2[:, :, index, :] / factors.reshape((1, 1, -1, 1))
        else:
            raise ValueError('Unsupported weight initializer: %s' % init)

        student_w1 = np.concatenate((teacher_w1, new_w1), axis=3)
        if init == 'random-pad':
            student_w2 = np.concatenate((teacher_w2, new_w2), axis=2)
        elif init == 'net2wider':
            # add small noise to break symmetry, so that student model will have
            # full capacity later
            noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
            student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=2)
            student_w2[:, :, index, :] = new_w2
        student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

        return student_w1, student_b1, student_w2


if __name__ == "__main__":

    dbg = False
    if dbg:
        config = Config(epochs=0, verbose=1, limit_data=9999)
    else:
        config = Config(epochs=100, verbose=1, limit_data=1)
    teacher_model = MyModel(config, [["Conv2D", 'conv1', {'filters': 64}],
                                     ["Conv2D", 'conv2', {'filters': 64}],
                                     ["Conv2D", 'conv3', {'filters': 10}],
                                     ['GlobalMaxPooling2D', 'gmpool1', {}],
                                     ['Activation', 'activation1', {'activation_type': 'softmax'}]])

    teacher_model.comp_fit_eval()

    net2net = Net2Net()

    student_model = net2net.wider_conv2d(teacher_model, name='conv2', width_ratio=2)
    # student_model = net2net.deeper_conv2d(teacher_model, name='conv2')
    student_model.comp_fit_eval()
    # from IPython import embed; embed()


class Legacy(object):
    def load_data(dbg=False):
        transfer_x, transfer_y = get_transfer_data(osp.join(root_dir, "data", "transfer_data"))
        transfer_y = transfer_y.reshape((-1, 1))
        (train_x, train_y), (test_x, test_y) = cifar10.load_data()
        train_x = np.concatenate((train_x, transfer_x))
        train_y = np.concatenate((train_y, transfer_y))
        # test_x = np.concatenate((test_x, transfer_x[:transfer_x.shape[0] * 7 / 10, ...]))
        # test_y = np.concatenate((test_y, transfer_y[:transfer_y.shape[0] * 7 / 10, ...]))

        '''Train and Test use the same mean img'''
        train_x, mean_image = preprocess_input(train_x, None)
        test_x, _ = preprocess_input(test_x, mean_image)

        train_y, test_y = map(preprocess_output, [train_y, test_y])

        train_logits = np.asarray(np.load(osp.join(root_dir, "data", "resnet18_logits_transfer.npy")))
        test_logits = np.asarray(np.load(osp.join(root_dir, "data", "resnet18_logits_test.npy")))
        print('train_logits.shape: ', train_logits.shape)
        print('test_logits.shape: ', test_logits.shape)

        if dbg:
            train_x, train_y, \
            test_x, test_y, \
            train_logits, test_logits \
                = map(lambda x: limit_data(x, division=9999), [train_x, train_y,
                                                               test_x, test_y,
                                                               train_logits, test_logits]
                      )
        else:
            '''For speedup'''
            train_x, train_y, \
            test_x, test_y, \
            train_logits, test_logits \
                = map(lambda x: limit_data(x, division=5), [train_x, train_y,
                                                            test_x, test_y,
                                                            train_logits, test_logits]
                      )
        # train_data = (train_x, [train_logits, train_y])
        # test_data = (test_x, [test_logits, test_y])
        print("\n\n---------------------------------------\n\n")

        train_data = (train_x, train_y)
        test_data = (test_x, test_y)
        check_data_format(train_data, test_data)

        return train_data, test_data

    # knowledge transfer algorithms
    def wider_conv2d_weight(teacher_w1, teacher_b1, teacher_w2, new_width, init, help=None):
        '''Get initial weights for a wider conv2d layer with a bigger nb_filter,
        by 'random-pad' or 'net2wider'.

        # Arguments
            teacher_w1: `weight` of conv2d layer to become wider,
              of shape (nb_filter1, nb_channel1, kh1, kw1)
            teacher_b1: `bias` of conv2d layer to become wider,
              of shape (nb_filter1, )
            teacher_w2: `weight` of next connected conv2d layer,
              of shape (nb_filter2, nb_channel2, kh2, kw2)
            new_width: new `nb_filter` for the wider conv2d layer
            init: initialization algorithm for new weights,
              either 'random-pad' or 'net2wider'
        '''
        # with open("debug.pkl", "w") as f:
        #     cPickle.dump([teacher_w1, teacher_b1, teacher_w2, new_width], f)
        if len(teacher_w1.shape) == 4 and len(teacher_w2.shape) == 2 and help is not None:
            n = new_width - teacher_w1.shape[0]
            index = np.random.randint(teacher_w1.shape[0], size=n)
            factors = np.bincount(index)[index] + 1
            new_w1 = teacher_w1[index, ...]
            noise = np.random.normal(0, 5e-2 * new_w1.std(), size=new_w1.shape)
            new_w_conv1 = np.concatenate([teacher_w1, new_w1 + noise], axis=0)
            new_b1 = teacher_b1[index]
            noise = np.random.normal(0, 5e-2 * new_b1.std(), size=new_b1.shape)
            new_b_conv1 = np.concatenate([teacher_b1, new_b1 + noise], axis=0)

            index_fc = np.empty(shape=[0, ], dtype=int)
            factor_fc = np.empty(shape=[0, ], dtype=int)
            for i, j in zip(index.reshape((-1,)), factors.reshape((-1,))):
                start = i * help
                end = (i + 1) * help
                #     print index_fc.shape,np.arange(start,end).shape
                index_fc = np.concatenate([index_fc, np.arange(start, end)])

                factor_fc = np.concatenate([factor_fc,
                                            j * np.ones(shape=(help,))])

            new_w2 = teacher_w2[index_fc, :] / factor_fc.reshape((-1, 1))
            noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
            new_w_fc1 = np.concatenate([teacher_w2, new_w2 + noise], axis=0)
            new_w_fc1[index_fc, :] = new_w2

            return new_w_conv1, new_b_conv1, new_w_fc1
        else:
            assert teacher_w1.shape[3] == teacher_w2.shape[2], (
                'successive layers from teacher model should have compatible shapes')
            assert teacher_w1.shape[3] == teacher_b1.shape[0], (
                'weight and bias from same layer should have compatible shapes')
            assert new_width > teacher_w1.shape[3], (
                'new width (filters) should be bigger than the existing one')

            n = new_width - teacher_w1.shape[3]
            if init == 'random-pad':
                new_w1 = np.random.normal(0, 0.1, size=teacher_w1.shape[:-1] + (n,))
                new_b1 = np.ones(n) * 0.1
                new_w2 = np.random.normal(0, 0.1, size=teacher_w2.shape[:2] + (
                    n, teacher_w2.shape[3]))
            elif init == 'net2wider':
                index = np.random.randint(teacher_w1.shape[3], size=n)
                factors = np.bincount(index)[index] + 1.
                new_w1 = teacher_w1[:, :, :, index]
                new_b1 = teacher_b1[index]
                new_w2 = teacher_w2[:, :, index, :] / factors.reshape((1, 1, -1, 1))
            else:
                raise ValueError('Unsupported weight initializer: %s' % init)

            student_w1 = np.concatenate((teacher_w1, new_w1), axis=3)
            if init == 'random-pad':
                student_w2 = np.concatenate((teacher_w2, new_w2), axis=2)
            elif init == 'net2wider':
                # add small noise to break symmetry, so that student model will have
                # full capacity later
                noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
                student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=2)
                student_w2[:, :, index, :] = new_w2
            student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

            return student_w1, student_b1, student_w2

    def wider_fc_weight(teacher_w1, teacher_b1, teacher_w2, new_width, init):
        '''Get initial weights for a wider fully connected (dense) layer
           with a bigger nout, by 'random-padding' or 'net2wider'.
    
        # Arguments
            teacher_w1: `weight` of fc layer to become wider,
              of shape (nin1, nout1)
            teacher_b1: `bias` of fc layer to become wider,
              of shape (nout1, )
            teacher_w2: `weight` of next connected fc layer,
              of shape (nin2, nout2)
            new_width: new `nout` for the wider fc layer
            init: initialization algorithm for new weights,
              either 'random-pad' or 'net2wider'
        '''

        assert teacher_w1.shape[1] == teacher_w2.shape[0], (
            'successive layers from teacher model should have compatible shapes')
        assert teacher_w1.shape[1] == teacher_b1.shape[0], (
            'weight and bias from same layer should have compatible shapes')
        assert new_width > teacher_w1.shape[1], (
            'new width (nout) should be bigger than the existing one')

        n = new_width - teacher_w1.shape[1]
        if init == 'random-pad':
            new_w1 = np.random.normal(0, 0.1, size=(teacher_w1.shape[0], n))
            new_b1 = np.ones(n) * 0.1
            new_w2 = np.random.normal(0, 0.1, size=(n, teacher_w2.shape[1]))
        elif init == 'net2wider':
            index = np.random.randint(teacher_w1.shape[1], size=n)
            factors = np.bincount(index)[index] + 1.
            new_w1 = teacher_w1[:, index]
            new_b1 = teacher_b1[index]
            new_w2 = teacher_w2[index, :] / factors[:, np.newaxis]
        else:
            raise ValueError('Unsupported weight initializer: %s' % init)

        student_w1 = np.concatenate((teacher_w1, new_w1), axis=1)
        if init == 'random-pad':
            student_w2 = np.concatenate((teacher_w2, new_w2), axis=0)
        elif init == 'net2wider':
            # add small noise to break symmetry, so that student model will have
            # full capacity later
            noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
            student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=0)
            student_w2[index, :] = new_w2
        student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

        return student_w1, student_b1, student_w2

    def deeper_conv2d_weight(teacher_w):
        '''Get initial weights for a deeper conv2d layer by net2deeper'.
    
        # Arguments
            teacher_w: `weight` of previous conv2d layer,
              of shape (nb_filter, nb_channel, kh, kw)
        '''
        kw, kh, num_channel, filters = teacher_w.shape
        student_w = np.zeros((kw, kh, filters, filters))
        for i in xrange(filters):
            student_w[(kw - 1) / 2, (kh - 1) / 2, i, i] = 1.
        student_b = np.zeros(filters)
        return student_w, student_b

    def copy_weights(teacher_model, student_model, layer_names=None):
        if layer_names is None:
            layer_names = [l.name for l in teacher_model.layers]
        for name in layer_names:
            weights = teacher_model.get_layer(name=name).get_weights()
            try:
                student_model.get_layer(name=name).set_weights(weights)
            except ValueError as e:
                # print("some layer shape change, Don't copy, We Init it Manually! OK!")
                # print("\t It is OK! Detail: {}".format(e))
                pass


                # methods to construct teacher_model and student_models
        '''Copy weights from teacher_model to student_model,
         for layers with names listed in layer_names
        '''

    def make_teacher_model(train_data, validation_data, nb_epoch, verbose):
        '''Train a simple CNN as teacher model.
        '''
        model = Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=input_shape,
                         padding='same', name='conv1', activation="relu"))
        model.add(MaxPooling2D(name='pool1'))
        # model.add(Dropout(0.25,name='conv_drop1'))
        model.add(Conv2D(64, (3, 3), padding='same', name='conv2', activation="relu"))
        model.add(MaxPooling2D(name='pool2'))
        # model.add(Dropout(0.25,name='conv_drop2'))
        model.add(Flatten(name='flatten'))
        model.add(Dense(64, activation='relu', name='fc1'))
        # model.add(Dropout(0.5,name="fc_drop1"))
        model.add(Dense(nb_class, activation='softmax', name='fc2'))
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.01, momentum=0.9),
                      metrics=['accuracy'])
        shuffle_weights(model)
        print([l.name for l in model.layers])
        # save_model_config(model,"sequential")
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=2,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,
            channel_shift_range=0.001)  # randomly flip images

        datagen.fit(train_data[0])
        history = model.fit_generator(datagen.flow(train_data[0], train_data[1],
                                                   batch_size=batch_size),
                                      steps_per_epoch=train_data[0].shape[0],
                                      epochs=nb_epoch,
                                      validation_data=validation_data,
                                      verbose=2, callbacks=[lr_reducer, early_stopper, csv_logger])
        # history = model.fit(
        #     *(train_data),
        #     nb_epoch=nb_epoch,
        #     validation_data=validation_data,
        #     verbose=verbose,
        #     callbacks=[lr_reducer, early_stopper, csv_logger],batch_size=batch_size
        # )

        return model, history

    def get_name_ind_map(student_model_dict):
        if student_model_dict["class_name"] == "Model":
            name2ind = {student_model_dict["config"]["layers"][i]["config"]["name"]: i
                        for i in range(len(student_model_dict["config"]["layers"]))}
            ind2name = [student_model_dict["config"]["layers"][i]["config"]["name"]
                        for i in range(len(student_model_dict["config"]["layers"]))]
        else:
            # print(student_model_dict["class_name"])
            assert student_model_dict["class_name"] == "Sequential"
            name2ind = {student_model_dict["config"][i]["config"]["name"]: i
                        for i in range(len(student_model_dict["config"]))}
            ind2name = [student_model_dict["config"][i]["config"]["name"]
                        for i in range(len(student_model_dict["config"]))]
        return name2ind, ind2name

    def get_width(model):
        mdict = json.loads(model.to_json())
        name2ind, ind2name = get_name_ind_map(mdict)
        student_conv_width = []
        student_fc_width = []
        for i, v in enumerate(ind2name):

            if mdict["config"][i]["class_name"] == "Conv2D":
                student_conv_width += [get_config(mdict, i, "filters")]
            elif mdict["config"][i]["class_name"] == "Dense":
                student_fc_width += [get_config(mdict, i, "units")]
        return student_conv_width, student_fc_width

    def get_config(dict, modify_layer, property):
        name2ind, ind2name = get_name_ind_map(dict)
        if isinstance(modify_layer, basestring):
            modify_layer_ind = name2ind[modify_layer]
        else:
            modify_layer_ind = modify_layer

        if dict["class_name"] == "Model":
            return dict["config"]["layers"][modify_layer_ind]["config"][property]
        else:
            # Sequential will be Deprecated
            assert dict["class_name"] == "Sequential"
            return dict["config"][modify_layer_ind]["config"][property]

    def set_config(dict, modify_layer, property, value):
        name2ind, ind2name = get_name_ind_map(dict)
        if isinstance(modify_layer, basestring):
            modify_layer_ind = name2ind[modify_layer]
        else:
            modify_layer_ind = modify_layer

        if dict["class_name"] == "Model":
            dict["config"]["layers"][modify_layer_ind]["config"][property] = value
        else:
            # Sequential will be Deprecated
            dict["config"][modify_layer_ind]["config"][property] = value

    def wider_conv2d_dict(new_conv1_width_ratio, modify_layer_name, teacher_model_dict):
        student_model_dict = copy.deepcopy(teacher_model_dict)
        # student_model_dict=teacher_model_dict

        CONV_WIDTH_LIMITS = 512
        old_conv1_width = get_config(student_model_dict, modify_layer_name, "filters")
        new_conv1_width = min(
            CONV_WIDTH_LIMITS,
            int(
                new_conv1_width_ratio
                * old_conv1_width
            )
        )
        set_config(student_model_dict, modify_layer_name, "filters", new_conv1_width)
        # No need:
        # student_model_dict["config"][modify_layer_ind+1]["config"]["batch_input_shape"][1]=new_conv1_width
        return student_model_dict, new_conv1_width, new_conv1_width == CONV_WIDTH_LIMITS or new_conv1_width == old_conv1_width

    def to_uniform_init_dict(student_model_dict):
        # need to be add when dropout needed
        # but I add it now
        for layer in student_model_dict["config"]:
            if layer["class_name"] == "Dense":
                layer["config"]["init"] = "glorot_uniform"
        return student_model_dict
        # pass

    def to_last_layer_softmax(dict):
        for layer in dict["config"][:-1]:
            if layer["class_name"] == "Dense":
                layer["config"]["activation"] = "relu"
        dict["config"][-1]["config"]["activation"] = "softmax"

    def wider_fc_dict(new_fc1_width_ratio, modify_layer_name, teacher_model_dict):
        student_model_dict = copy.deepcopy(teacher_model_dict)
        # student_model_dict=teacher_model_dict
        FC_WIDTH_LIMITS = 4096
        old_fc1_width = get_config(student_model_dict, modify_layer_name, "units")
        new_fc1_width = min(
            FC_WIDTH_LIMITS,
            int(new_fc1_width_ratio * (old_fc1_width))
        )
        set_config(student_model_dict, modify_layer_name, "units", new_fc1_width)
        # set_config(student_model_dict, modify_layer_name, "init", "glorot_uniform")
        # student_model_dict["config"][modify_layer_ind+1]["config"]["init"] = "glorot_uniform"
        # student_model_dict = to_uniform_init_dict(student_model_dict)
        ## automated!
        ## student_model_dict["config"][modify_layer_ind+1]["config"]["input_dim"]=new_fc1_width
        # student_model_dict["config"][modify_layer_ind + 1]["config"]["batch_input_shape"]=[None,new_fc1_width]
        return student_model_dict, new_fc1_width, new_fc1_width == FC_WIDTH_LIMITS or old_fc1_width == new_fc1_width

    ## not being use
    # def reorder_dict(d):
    #     len_of_d = len(d["config"])
    #     name2ind = {d["config"][i]["config"]["name"]: i
    #                 for i in range(len(d["config"]))}
    #     names = [d["config"][i]["config"]["name"]
    #              for i in range(len(d["config"]))]
    #
    #     for i, v in enumerate(names):
    #         if re.findall(r"conv\d+_\d+", v) or \
    #                 re.findall(r"fc\d+_\d+", v):
    #             break
    #     type = re.findall(r"[a-z]+", names[i])[0]
    #     # ind=int(re.findall(r"\d+",names[i])[0])
    #     start = 1
    #     for i, v in enumerate(names):
    #         now_type = re.findall(r"[a-z]+", v)[0]
    #         if now_type == type:
    #             names[i] = type + str(start)
    #             start += 1
    #     for i in range(len_of_d):
    #         d["config"][i]["config"]["name"] = names[i]
    #     return d


    def reorder_list(layers):
        name2ind = {v: i for i, v in enumerate(layers)}
        for i, v in enumerate(layers):
            # if re.findall(r"conv\d+_\d+", v) or \
            #         re.findall(r"fc\d+_\d+", v):
            #     break
            if v[0:2] == "^_":
                break
        layer_type = re.findall(r"[a-z]+", layers[i])[0]
        assert layer_type == "conv" or layer_type == "fc"
        start = 1
        for i, v in enumerate(layers):
            now_layer_type = re.findall(r"[a-z]+", v)[0]
            if now_layer_type == layer_type:
                layers[i] = layer_type + str(start)
                start += 1
        return layers

    def reorder_model(model):
        names = [l.name for l in model.layers]
        len_of_model = len(names)
        names2ind = {v: i for i, v in enumerate(names)}

        for i, v in enumerate(names):
            # if re.findall(r"conv\d+_\d+", v) or \
            #         re.findall(r"fc\d+_\d+", v):
            #     break
            if v[0:2] == "^_":
                break
        type = re.findall(r"[a-z]+", names[i])[0]
        assert type == "conv" or type == "fc"
        start = 1
        for i, v in enumerate(names):
            now_type = re.findall(r"[a-z]+", v)[0]
            if now_type == type:
                names[i] = type + str(start)
                start += 1
        for i, v in enumerate(names):
            model.layers[i].name = v
        return model

    def make_wider_student_model(teacher_model,
                                 train_data, validation_data,
                                 init, new_name, new_width_ratio, nb_epoch=3, verbose=1):
        '''Train a wider student model based on teacher_model,
           with either 'random-pad' (baseline) or 'net2wider'
        '''
        print([l.name for l in teacher_model.layers])
        new_type = re.findall(r"[a-z]+", new_name)[0]
        assert new_type == "conv" or new_type == "fc"
        new_ind = re.findall(r"\d+", new_name)[0]
        new_ind = int(new_ind)
        teacher_model_dict = json.loads(teacher_model.to_json())
        if new_type == "conv":

            new_conv1_name = new_name
            student_model_dict, new_conv1_width, return_flag \
                = wider_conv2d_dict(new_width_ratio, new_conv1_name,
                                    teacher_model_dict)

        elif new_type == "fc":

            new_fc1_name = new_name
            student_model_dict, new_fc1_width, return_flag \
                = wider_fc_dict(new_width_ratio, new_fc1_name,
                                teacher_model_dict)
        if return_flag:
            return teacher_model, None
        model = model_from_json(json.dumps(student_model_dict))

        layer_name = [l.name for l in teacher_model.layers]
        # layer_name=set(layer_name)
        # layer_name.discard(new_conv1_name)
        # layer_name.discard(new_fc1_name)
        copy_weights(teacher_model, model, layer_name)

        if new_type == "conv":
            next_new_name = new_type + str(new_ind + 1)
            if next_new_name not in [l.name for l in model.layers]:
                next_new_name = "fc1"
                w_conv1, b_conv1 = teacher_model.get_layer(new_name).get_weights()
                w_conv2, b_conv2 = teacher_model.get_layer(next_new_name).get_weights()
                from operator import mul
                print(reduce(mul, model.get_layer("pool2").get_output_shape_at(0)[-2:], 1))
                new_w_conv1, new_b_conv1, new_w_conv2 = wider_conv2d_weight(
                    w_conv1, b_conv1, w_conv2, new_conv1_width, init,
                    reduce(mul, model.get_layer("pool2").get_output_shape_at(0)[-2:], 1))
                # TODO consider pool dropout
            else:
                w_conv1, b_conv1 = teacher_model.get_layer(new_name).get_weights()
                w_conv2, b_conv2 = teacher_model.get_layer(next_new_name).get_weights()
                new_w_conv1, new_b_conv1, new_w_conv2 = wider_conv2d_weight(
                    w_conv1, b_conv1, w_conv2, new_conv1_width, init)

            model.get_layer(new_name).set_weights([new_w_conv1, new_b_conv1])
            model.get_layer(next_new_name).set_weights([new_w_conv2, b_conv2])

        elif new_type == "fc":
            next_new_name = new_type + str(new_ind + 1)
            w_fc1, b_fc1 = teacher_model.get_layer(new_name).get_weights()
            w_fc2, b_fc2 = teacher_model.get_layer(next_new_name).get_weights()
            new_w_fc1, new_b_fc1, new_w_fc2 = wider_fc_weight(
                w_fc1, b_fc1, w_fc2, new_fc1_width, init)
            model.get_layer(new_name).set_weights([new_w_fc1, new_b_fc1])
            model.get_layer(next_new_name).set_weights([new_w_fc2, b_fc2])

        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.001, momentum=0.9),
                      metrics=['accuracy'])
        print([l.name for l in model.layers])
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=2,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,
            channel_shift_range=0.001)  # randomly flip images

        datagen.fit(train_data[0])
        history = model.fit_generator(datagen.flow(train_data[0], train_data[1],
                                                   batch_size=batch_size),
                                      steps_per_epoch=train_data[0].shape[0],
                                      epochs=nb_epoch,
                                      validation_data=validation_data,
                                      verbose=2, callbacks=[lr_reducer, early_stopper, csv_logger])
        # history = model.fit(
        #     *(train_data),
        #     nb_epoch=nb_epoch,
        #     validation_data=validation_data,
        #     verbose=verbose if nb_epoch != 0 else 0,
        #     callbacks=[lr_reducer, early_stopper, csv_logger],batch_size=batch_size
        # )

        return model, history

    def make_deeper_student_model(teacher_model,
                                  train_data,
                                  validation_data,
                                  init, new_name, nb_epoch=3, verbose=1):
        '''Train a deeper student model based on teacher_model,
           with either 'random-init' (baseline) or 'net2deeper'
        '''
        print([l.name for l in teacher_model.layers])
        new_type = re.findall(r"[a-z]+", new_name)[0]
        new_ind = int(re.findall(r"\d+", new_name)[0])

        teacher_model_dict = json.loads(teacher_model.to_json())
        student_model_dict = copy.deepcopy(teacher_model_dict)
        name2ind, ind2name = get_name_ind_map(student_model_dict)

        student_new_layer = copy.deepcopy(student_model_dict["config"][name2ind[new_name]])
        student_new_layer["config"]["name"] = "^_" + new_name
        if new_type == "conv":
            # student_new_layer["config"]["filters"] # NO NEED
            pass
        elif new_type == "fc":
            student_new_layer["config"]["init"] = "identity"
        student_model_dict["config"].insert(name2ind[new_name] + 1, student_new_layer)
        to_last_layer_softmax(student_model_dict)
        model = model_from_json(json.dumps(student_model_dict))

        if new_type == "conv" and init == 'net2deeper':
            prev_w, _ = teacher_model.get_layer(new_name).get_weights()
            new_weights = deeper_conv2d_weight(prev_w)
            model.get_layer("^_" + new_name).set_weights(new_weights)

        ## copy weights for other layers
        copy_weights(teacher_model, model)
        # print([l.name for l in model.layers])
        model = reorder_model(model)
        # print([l.name for l in model.layers])
        model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.001, momentum=0.9),
                      metrics=['accuracy'])
        print([l.name for l in model.layers])
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=2,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False,
            channel_shift_range=0.001)  # randomly flip images
        datagen.fit(train_data[0])
        history = model.fit_generator(datagen.flow(train_data[0], train_data[1],
                                                   batch_size=batch_size),
                                      samples_per_epoch=train_data[0].shape[0],
                                      nb_epoch=nb_epoch,
                                      validation_data=validation_data,
                                      verbose=2, callbacks=[lr_reducer, early_stopper, csv_logger])
        # history = model.fit(
        #     *(train_data),
        #     nb_epoch=nb_epoch,
        #     validation_data=validation_data,
        #     verbose=verbose if nb_epoch != 0 else 0,
        #     callbacks=[lr_reducer, early_stopper, csv_logger],batch_size=batch_size
        # )

        return model, history

    def copy_model(model):
        model_dict = json.loads(model.to_json())
        new_model = model_from_json(json.dumps(model_dict))
        copy_weights(model, new_model)
        new_model.compile(loss='categorical_crossentropy',
                          optimizer=SGD(lr=0.001, momentum=0.9),
                          metrics=['accuracy'])
        for a, b in zip(new_model.get_weights(), model.get_weights()):
            test = a == b
            assert test.all()
        assert model is not new_model
        return new_model

    def make_model(teacher_model, commands, train_data, validation_data):
        student_model = copy_model(teacher_model)
        # student_model = shuffle_weights(student_model)
        log = []
        print("\n------------------------------\n")

        for cmd in commands[1:]:
            print("\n------------------------------\n")
            os.system("nvidia-smi")
            print("Attention: ", cmd)
            print(get_width(student_model))
            # vis_utils.plot_model(student_model, to_file=str(int(time.time())) + '.pdf', show_shapes=True)
            if cmd[0] == "net2wider" or cmd[0] == "random-pad":
                student_model, history = make_wider_student_model(
                    student_model,
                    train_data, validation_data,
                    *cmd
                )
            elif cmd[0] == "net2deeper" or cmd[0] == "random-init":
                student_model, history = make_deeper_student_model(
                    student_model,
                    train_data, validation_data,
                    *cmd
                )
            else:
                raise ValueError('Unsupported cmd: %s' % cmd[0])
            student_model.summary()
            os.system("nvidia-smi")
            print(get_width(student_model))
            if history == None:
                continue
            # print(raw_input("check mem"))
            log_append_t = [
                cmd,
                [l.name for l in student_model.layers],
                history.history["val_acc"] if history.history else []
            ]

            log.append(log_append_t)

        name = commands[0]

        save_model_config(student_model, name)

        _shell_cmd = "mkdir -p " + osp.join(root_dir, "output", name)

        subprocess.call(_shell_cmd.split())
        os.chdir(osp.join(root_dir, "output", name))

        with open(name + ".pkl", 'w') as f:
            cPickle.dump(log, f)
        with open(name + ".log", "w") as f:
            for log_item in log:
                f.write(str(log_item[0]) + "\n")
                [f.write("\t" + str(log_item_item) + "\n") for log_item_item in log_item[1:]]
        vis_utils.plot_model(teacher_model, to_file="teacher.png", show_shapes=True)

        os.chdir("../../src")

        return student_model, log

    def save_model_config(student_model, name):
        _shell_cmd = "mkdir -p " + osp.join(root_dir, "output", name)
        subprocess.call(_shell_cmd.split())
        os.chdir(osp.join(root_dir, "output", name))
        student_model.save_weights(name + ".h5")
        with open(name + ".json", "w") as f:
            json.dump(
                json.loads(student_model.to_json()),
                f,
                indent=2
            )
        vis_utils.plot_model(student_model, to_file=name + '.pdf', show_shapes=True)
        os.chdir("../../src")

    ## a good example to combine tf
    def reset_weights(model):
        session = K.get_session()
        for layer in model.layers:
            if isinstance(layer, Dense):
                old = layer.get_weights()
                layer.W.initializer.run(session=session)
                layer.b.initializer.run(session=session)
                print(np.array_equal(old, layer.get_weights()), " after initializer run")
            else:
                print(layer, "not reinitialized")

    def shuffle_weights(model, weights=None):
        """Randomly permute the weights in `model`, or the given `weights`.
    
        This is a fast approximation of re-initializing the weights of a model.
    
        Assumes weights are distributed independently of the dimensions of the weight tensors
          (i.e., the weights have the same distribution along each dimension).
    
        :param Model model: Modify the weights of the given model.
        :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
          If `None`, permute the model's current weights.
        """
        if weights is None:
            weights = model.get_weights()
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
        # Faster, but less random: only permutes along the first dimension
        # weights = [np.random.permutation(w) for w in weights]
        model.set_weights(weights)
        return model

    ## will be replace by tensorboard
    # def smooth(x, y):
    #     import matplotlib.pyplot as plt
    #     from scipy.optimize import curve_fit
    #     from scipy.interpolate import interp1d
    #     from scipy.signal import savgol_filter
    #     import numpy as np
    #     x, y = np.array(x), np.array(y)
    #     xx = np.linspace(x.min(), x.max(), x.shape[0] + 100)
    #     # xx=x
    #     # interpolate + smooth
    #     itp = interp1d(x, y, kind='linear')
    #     window_size, poly_order = 101, 3
    #     yy_sg = savgol_filter(itp(xx), window_size, poly_order)
    #     return xx, yy_sg

    def vis(log0, log12, command):
        acc0 = [0] + log0[0][-1]
        plt.clf()
        plt.close("all")
        plt.figure()
        # plt.hold(True)
        for ind, log in enumerate(log12):
            acc = acc0
            plt.plot(np.arange(start=0, stop=len(acc)), np.array(acc))
            for log_item in log:
                plt.plot(np.arange(start=len(acc), stop=len(acc + log_item[-1])), np.array(log_item[-1]))
                acc += log_item[-1]
            acc = np.array(acc)
            print(acc.shape)
            np.save("val_acc" + str(ind) + ".npy", acc)
        # plt.legend([command[0]])
        _shell_cmd = "mkdir -p " + osp.join(root_dir, "output", command[0])
        subprocess.call(_shell_cmd.split())
        os.chdir(osp.join(root_dir, "output", command[0]))

        plt.savefig('val_acc.pdf')
        plt.savefig('val_acc.png')
        # try:
        #     plt.show()
        # finally:
        #     pass
        os.chdir(osp.join(root_dir, "src"))
