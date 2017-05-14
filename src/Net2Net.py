# !/usr/bin/env python
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy,subprocess,pprint
import scipy.ndimage
import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10
from keras.utils import np_utils

from Init import root_dir
from Log import logger
from Model import MyModel,MyGraph
from Config import Config

class Net2Net(object):

    def wider_conv2d(self, model, name, width_ratio):
        # modify graph
        new_graph=model.graph.copy()
        new_node=new_graph.get_nodes(name)[0]
        assert new_node.type=='Conv2D','must wide a conv'
        new_width=new_node.config['filters']*width_ratio
        new_node.config['filters']=new_width

        logger.debug(new_graph.to_json())
        # construct model
        new_model=MyModel(config=model.config,graph= new_graph)

        # inherit weight
        w_conv1, b_conv1 = model.get_layers(name)[0].get_weights()
        w_conv2, b_conv2 = model.get_layers(name, next_layer=True)[0].get_weights()

        new_w_conv1, new_b_conv1, new_w_conv2 = self._wider_conv2d_weight(
            w_conv1, b_conv1, w_conv2, new_width, "net2wider")

        new_model.get_layers(name)[0].set_weights([new_w_conv1, new_b_conv1])
        new_model.get_layers(name, next_layer=True)[0].set_weights([new_w_conv2, b_conv2])

        return new_model

    def deeper_conv2d(self, model, name, kernel_size=3, filters='same'):
        # construct graph
        new_graph = model.graph.copy()
        from Model import Node

        new_node=Node(type='Conv2D',name='new',
                     config= {'kernel_size':kernel_size,'filters':filters}
                      )
        new_graph.deeper(name,new_node)
        logger.debug(new_graph.to_json())

        # construc model
        new_model=MyModel(config=model.config,graph=new_graph)

        # inherit weight
        w_conv1, b_conv1 = model.get_layers(name, last_layer=True).get_weights()
        w_conv2, b_conv2 = model.get_layers(name).get_weights()

        new_w_conv2, new_b_conv2 = self._deeper_conv2d_weight(
            w_conv1, b_conv1, w_conv2, b_conv2, "net2deeper")

        new_model.get_layers(name).set_weights([new_w_conv2, new_b_conv2])

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

    dbg = True
    if dbg:
        config = Config(epochs=0, verbose=1, limit_data=9999)
    else:
        config = Config(epochs=100, verbose=1, limit_data=1)
    model_l = [["Conv2D", 'conv1', {'filters': 64}],
               ["Conv2D", 'conv2', {'filters': 64}],
               ["Conv2D", 'conv3', {'filters': 10}],
               ['GlobalMaxPooling2D', 'gmpool1', {}],
               ['Activation', 'activation1', {'activation_type': 'softmax'}]]
    graph = MyGraph(model_l)
    teacher_model = MyModel(config, graph)

    teacher_model.comp_fit_eval()

    net2net = Net2Net()

    # student_model = net2net.wider_conv2d(teacher_model, name='conv2', width_ratio=2)
    student_model = net2net.deeper_conv2d(teacher_model, name='conv2')
    student_model.comp_fit_eval()
    # from IPython import embed; embed()
