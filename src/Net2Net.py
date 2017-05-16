# !/usr/bin/env python
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
import scipy.ndimage
import keras

import Utils
from Config import Config
from Log import logger
from Model import MyModel, MyGraph, Node


class Net2Net(object):
    def skip(self, model, from_name, to_name):
        new_graph = model.graph.copy()
        node1 = new_graph.get_nodes(from_name)[0]
        node2 = new_graph.get_nodes(to_name)[0]
        node3 = new_graph.get_nodes(to_name, next_layer=True)[0]
        new_graph.update()
        new_name = 'Concatenate' + \
                   str(
                       1 + max(new_graph.type2ind.get('Concatenate', [0]))
                   )
        new_node = Node(type='Concatenate', name=new_name, config={})  # we use channel first, will set axis = 1 latter
        new_graph.add_node(new_node)
        new_graph.remove_edge(node2, node3)
        new_graph.add_edge(node1, new_node)
        new_graph.add_edge(node2, new_node)
        new_graph.add_edge(new_node, node3)
        logger.debug(new_graph.to_json())
        new_model = MyModel(config=model.config, graph=new_graph)

        # handle weight

        return new_model

    def wider_conv2d(self, model, name, width_ratio):
        # modify graph
        new_graph = model.graph.copy()
        new_node = new_graph.get_nodes(name)[0]
        assert new_node.type == 'Conv2D', 'must wide a conv'
        new_width = new_node.config['filters'] * width_ratio
        new_node.config['filters'] = new_width

        logger.debug(new_graph.to_json())
        # construct model
        new_model = MyModel(config=model.config, graph=new_graph)

        # inherit weight
        w_conv1, b_conv1 = model.get_layers(name)[0].get_weights()
        w_conv2, b_conv2 = model.get_layers(name, next_layer=True)[0].get_weights()

        new_w_conv1, new_b_conv1, new_w_conv2 = Net2Net._wider_conv2d_weight(
            w_conv1, b_conv1, w_conv2, new_width, "net2wider")

        new_model.get_layers(name)[0].set_weights([new_w_conv1, new_b_conv1])
        new_model.get_layers(name, next_layer=True)[0].set_weights([new_w_conv2, b_conv2])

        return new_model

    def deeper_conv2d(self, model, name, kernel_size=3, filters='same'):
        # construct graph
        new_graph = model.graph.copy()

        new_node = Node(type='Conv2D', name='new',
                        config={'kernel_size': kernel_size, 'filters': filters}
                        )
        new_graph.deeper(name, new_node)
        logger.debug(new_graph.to_json())

        # construc model
        new_model = MyModel(config=model.config, graph=new_graph)

        # inherit weight
        w_conv1, b_conv1 = new_model.get_layers(name)[0].get_weights()
        w_conv2, b_conv2 = new_model.get_layers(name, next_layer=True)[0].get_weights()

        new_w_conv2, new_b_conv2 = Net2Net._deeper_conv2d_weight(
            w_conv1, b_conv1, w_conv2, b_conv2, "net2deeper")

        new_model.get_layers(name, next_layer=True)[0].set_weights([new_w_conv2, new_b_conv2])

        return new_model

    @staticmethod
    def _deeper_conv2d_weight(teacher_w1, teacher_b1, teacher_w2, teacher_b2, init='net2deeper'):
        student_w, student_b = \
            Net2Net._convert_weight(teacher_w1, teacher_w2.shape), \
            Net2Net._convert_weight(teacher_b1, teacher_b2.shape)
        return student_w, student_b

    @staticmethod
    def _convert_weight(weight, nw_size):
        w_size = weight.shape
        assert len(w_size) == len(nw_size)
        w_ratio = [_nw / _w for _nw, _w in zip(nw_size, w_size)]
        new_weight = scipy.ndimage.zoom(weight, w_ratio)
        return new_weight

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
    model_l = [["Conv2D", 'Conv2D1', {'filters': 16}],
               ["Conv2D", 'Conv2D2', {'filters': 64}],
               ["Conv2D", 'Conv2D3', {'filters': 10}],
               ['GlobalMaxPooling2D', 'GlobalMaxPooling2D1', {}],
               ['Activation', 'Activation1', {'activation_type': 'softmax'}]]
    graph = MyGraph(model_l)
    ori_model = MyModel(config, graph,name='origin')
    Utils.vis_model(ori_model, 'origin')
    # Utils.vis_graph(teacher_model.graph,'origin')
    # ori_model.model.summary()
    ori_model.comp_fit_eval()

    trainable_count, non_trainable_count =  Utils.count_weight(ori_model)
    print(trainable_count,non_trainable_count)

    net2net = Net2Net()
    later_model = MyModel(config=config, model=ori_model)
    later_model = net2net.wider_conv2d(later_model, name='Conv2D2', width_ratio=2)
    later_model = net2net.deeper_conv2d(later_model, name='Conv2D2')
    later_model = net2net.skip(later_model, from_name='Conv2D1', to_name='Conv2D2')
    # ori_model.model.summary()
    Utils.vis_model(later_model, 'later')
    trainable_count, non_trainable_count = Utils.count_weight(later_model)
    print(trainable_count, non_trainable_count)
    # Utils.vis_graph(student_model.graph, 'later')
    later_model.set_name('later')

    later_model.comp_fit_eval()
    # from IPython import embed; embed()
