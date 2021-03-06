# !/usr/bin/env python
# Dependence: Model,Config
from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.utils.conv_utils import convert_kernel
import keras
import os

import networkx as nx
import numpy as np
import scipy
import scipy.ndimage

from Config import MyConfig
from Logger import logger
from Model import MyModel, MyGraph, Node

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from keras.models import Model


class Net2Net(object):
    @staticmethod
    def my_model2model(model):
        if isinstance(model, MyModel):
            return model.model
        else:
            return model

    def copy_model(self, model, config):
        from keras.utils.generic_utils import get_custom_objects
        from Model import IdentityConv, GroupIdentityConv

        get_custom_objects()['IdentityConv'] = IdentityConv
        get_custom_objects()['GroupIdentityConv'] = GroupIdentityConv

        new_model = MyModel(config, model.graph.copy(), keras.models.load_model(model.config.model_path))
        keras.models.save_model(new_model.model, new_model.config.model_path)
        return new_model

    def deeper(self, model, config, with_pooling):
        # select
        names = [node.name
                 for node in model.graph.nodes()
                 if node.type == 'Conv2D' or node.type == 'Group' or node.type == 'Conv2D_Pooling']
        while True:
            choice = names[np.random.randint(0, len(names))]
            next_nodes = model.graph.get_nodes(choice, next_layer=True, last_layer=False)
            if 'GlobalMaxPooling2D' not in [node.type for node in next_nodes]:
                break
        if not with_pooling:
            logger.info('choose {} to deeper'.format(choice))
            return self.deeper_conv2d(model, choice, kernel_size=3, filters='same', config=config, with_pooling=False), True
        else:
            logger.info('choose {} to deeper_with_pooling'.format(choice))
            return self.deeper_conv2d(model, choice, kernel_size=3, filters='same', config=config, with_pooling=True), True

    # find two conjacent conv layers, conv -> group is not allowed, only conv -> conv layer is okay
    # TODO: conv -> maxpooling -> conv, maxpooling will missing, same problem exists in deeper operation
    def wider(self, model, config):
        topo_nodes = nx.topological_sort(model.graph)
        names = [node.name
                 for node in topo_nodes
                 if
                 node.type == 'Conv2D' or node.type == 'Conv2D_Pooling' or node.type == 'Group']  # support group layer to wider
                 #node.type == 'Conv2D' or node.type == 'Conv2D_Pooling']
        max_iter = 100
        for i in range(max_iter + 1):
            if i == max_iter:
                logger.info('can\'t find a suitable layer to apply wider operation,return origin model')
                return model, False
            # random choose a layer to wider, except last conv layer
            choice = names[np.random.randint(0, len(names) - 1)]
            cur_node = model.graph.get_nodes(choice)[0]
            next_nodes = model.graph.get_nodes(choice, next_layer=True, last_layer=False)
            if 'Conv2D' in [node.type for node in next_nodes] or 'Conv2D_Pooling' in [node.type for node in next_nodes]:
                break
            else:
                continue

        cur_width = cur_node.config['filters']

        # for test
        # enlarge the max_cur_width
        #max_cur_width = (int((config.model_max_conv_width - config.model_min_conv_width) * cur_node.depth / config.model_max_depth) \
        #                + config.model_min_conv_width) * 5

        # for test
        max_cur_width = 1024

        width_ratio = np.random.rand()
        new_width = int(cur_width + width_ratio * (max_cur_width - cur_width))
        if cur_node.type == 'Group':
            # make sure that new_width % group_num == 0
            new_width = new_width // cur_node.config['group_num'] * cur_node.config['group_num']

        if new_width <= cur_width:
            logger.info('{} layer\'s width up to limit!'.format(choice))
            return model, False
        logger.info('choose {} to wider'.format(choice))
        if cur_node.type == 'Group':
            return self.wider_group_conv2d(model, layer_name=choice, new_width=new_width, config=config), True
        else:
            return self.wider_conv2d(model, layer_name=choice, new_width=new_width, config=config), True

    '''
        TODO: random choose two layer to skip 
    '''

    def add_skip(self, model, config):
        assert nx.is_directed_acyclic_graph(model.graph)
        topo_nodes = nx.topological_sort(model.graph)

        names = [node.name for node in topo_nodes
                 if node.type == 'Conv2D' or node.type == 'Group' or node.type == 'Conv2D_Pooling']

        if len(names) <= 2:
            logger.info('can\'t find a suitable layer to apply add_skip operation,return origin model')
            return model, False

        max_iter = 100
        for i in range(max_iter + 1):
            if i == max_iter:
                logger.info('can\'t find a suitable layer to apply add_skip operation,return origin model')
                return model, False
            from_idx = np.random.randint(0, len(names) - 2)
            to_idx = from_idx + 1
            next_nodes = model.graph.get_nodes(names[to_idx], next_layer=True, last_layer=False)
            if 'Add' in [node.type for node in next_nodes]:
                continue
            else:
                break

        from_name = names[from_idx]
        to_name = names[to_idx]
        logger.info('choose {} and {} to add_skip'.format(from_name, to_name))
        return self.skip(model, from_name, to_name, config), True

    # add group operation
    def add_group(self, model, config):
        topo_nodes = nx.topological_sort(model.graph)
        names = [node.name
                 for node in topo_nodes
                 if node.type == 'Conv2D' or node.type == 'Group' or node.type == 'Conv2D_Pooling']

        # random choose a layer to concat a group operation
        choice = names[np.random.randint(0, len(names) - 1)]
        group_num = np.random.randint(1, 5)  # group number: [1, 2, 3, 4]

        logger.info('choose {} to add group, group_num is {}'.format(choice, group_num))

        # add node and edge to the graph
        new_graph = model.graph.copy()
        node = new_graph.get_nodes(choice)[0]
        node_suc = new_graph.get_nodes(choice, next_layer=True)[0]
        new_graph.update()
        new_name = 'Group' + \
                   str(
                       1 + max(new_graph.type2ind.get('Group', [0]))
                   )

        filters = node.config['filters']
        new_node = Node(type='Group', name=new_name, config={'group_num': group_num, 'filters': filters})
        new_graph.add_node(new_node)
        new_graph.remove_edge(node, node_suc)
        new_graph.add_edge(node, new_node)
        new_graph.add_edge(new_node, node_suc)
        new_model = MyModel(config=config, graph=new_graph)

        self.copy_weight(model, new_model)
        return new_model, True

    def maxpool_by_name(self, model, name, config):
        new_graph = model.graph.copy()
        node = new_graph.get_nodes(name)[0]
        node1 = new_graph.get_nodes(name, next_layer=True)[0]

        new_graph.update()
        new_name = 'MaxPooling2D' + \
                   str(
                       1 + max(new_graph.type2ind.get('MaxPooling2D', [0]))
                   )
        new_node = Node(type='MaxPooling2D', name=new_name, config={})
        new_graph.add_node(new_node)
        new_graph.remove_edge(node, node1)
        new_graph.add_edge(node, new_node)
        new_graph.add_edge(new_node, node1)
        # logger.debug(new_graph.to_json())
        new_model = MyModel(config=config, graph=new_graph)
        self.copy_weight(model, new_model)
        return new_model

    def avepool_by_name(self, model, name, config):
        new_graph = model.graph.copy()
        node = new_graph.get_nodes(name)[0]
        node1 = new_graph.get_nodes(name, next_layer=True)[0]

        new_graph.update()
        new_name = 'AveragePooling2D' + \
                   str(
                       1 + max(new_graph.type2ind.get('AveragePooling2D', [0]))
                   )
        new_node = Node(type='AveragePooling2D', name=new_name, config={})
        new_graph.add_node(new_node)
        new_graph.remove_edge(node, node1)
        new_graph.add_edge(node, new_node)
        new_graph.add_edge(new_node, node1)
        # logger.debug(new_graph.to_json())
        new_model = MyModel(config=config, graph=new_graph)
        self.copy_weight(model, new_model)
        return new_model

    def copy_weight(self, before_model, after_model):

        _before_model = self.my_model2model(model=before_model)
        _after_model = self.my_model2model(model=after_model)

        layer_names = [l.name for l in _before_model.layers if
                       'input' not in l.name.lower() and
                       'maxpooling2d' not in l.name.lower() and
                       'add' not in l.name.lower() and
                       'dropout' not in l.name.lower() and
                       'p_re_lu' not in l.name.lower() and
                       'concatenate' not in l.name.lower()]
        for name in layer_names:
            weights = _before_model.get_layer(name=name).get_weights()
            try:
                _after_model.get_layer(name=name).set_weights(weights)
            except Exception as inst:
                logger.warning("ignore copy layer {} from model {} to model {} because {}".format(name,
                                                                                                  before_model.config.name,
                                                                                                  after_model.config.name,
                                                                                                  inst))

    def skip(self, model, from_name, to_name, config):
        # original: node1-> node2 -> node3
        # now: node1  ->  node2 ->  new_node -> node3
        #         -------------------->
        new_graph = model.graph.copy()
        node1 = new_graph.get_nodes(from_name)[0]
        node2 = new_graph.get_nodes(to_name)[0]
        node3 = new_graph.get_nodes(to_name, next_layer=True)[0]
        new_graph.update()
        new_name = 'Add' + \
                   str(
                       1 + max(new_graph.type2ind.get('Add', [0]))
                   )
        new_node = Node(type='Add', name=new_name, config={})
        new_graph.add_node(new_node)
        new_graph.remove_edge(node2, node3)
        new_graph.add_edge(node1, new_node)
        new_graph.add_edge(node2, new_node)
        new_graph.add_edge(new_node, node3)
        # logger.debug(new_graph.to_json())
        new_model = MyModel(config=config, graph=new_graph)
        self.copy_weight(model, new_model)

        # Way 1
        # w, b = new_model.get_layers(node2.name)[0].get_weights()
        # w, b = Net2Net.rand_weight_like(w)
        # new_model.get_layers(node2.name)[0].set_weights((w, b))

        # way 2 do nothing

        # way 3 divided by 2
        # w, b = new_model.get_layers(node2.name)[0].get_weights()
        # w, b = Net2Net.rand_weight_like(w)
        # w = w / 2
        # b = b / 2
        # new_model.get_layers(node2.name)[0].set_weights((w, b))

        return new_model

    def wider_conv2d(self, model, layer_name, new_width, config):
        # modify graph
        new_graph = model.graph.copy()
        new_node = new_graph.get_nodes(layer_name)[0]
        assert new_node.type == 'Conv2D' or new_node.type == 'Conv2D_Pooling', 'must wide a conv'
        new_node.config['filters'] = new_width

        # logger.debug(new_graph.to_json())
        # construct model
        new_model = MyModel(config=config, graph=new_graph)
        self.copy_weight(model, new_model)
        # inherit weight
        w_conv1, b_conv1 = model.get_layers(layer_name)[0].get_weights()
        w_conv2, b_conv2 = model.get_layers(layer_name, next_layer=True, type=['Conv2D', 'Conv2D_Pooling'])[
            0].get_weights()

        new_w_conv1, new_b_conv1, new_w_conv2 = Net2Net._wider_conv2d_weight(
            w_conv1, b_conv1, w_conv2, new_width, "net2wider")

        new_model.get_layers(layer_name)[0].set_weights([new_w_conv1, new_b_conv1])
        new_model.get_layers(layer_name, next_layer=True, type=['Conv2D', 'Conv2D_Pooling'])[0].set_weights(
            [new_w_conv2, b_conv2])

        return new_model

    def wider_group_conv2d(self, model, layer_name, new_width, config):
        # modify graph
        new_graph = model.graph.copy()
        node = new_graph.get_nodes(layer_name)[0]
        assert node.type == 'Group', 'must wide a group layer'
        node.config['filters'] = new_width

        new_model = MyModel(config=config, graph=new_graph)
        self.copy_weight(model, new_model)

        filter_num = int(new_width / node.config['group_num'])

        for i in range(node.config['group_num']):
            if i == node.config['group_num'] - 1:  # last group
                filter_num = new_width - i * int((new_width / node.config['group_num']))

            layer_name_conv_1 = layer_name + '_conv2d_' + str(i) + '_1'
            layer_name_conv_2 = layer_name + '_conv2d_' + str(i) + '_2'

            # use keras get_layer function
            w_conv1, b_conv1 = model.model.get_layer(layer_name_conv_1).get_weights()
            w_conv2, b_conv2 = model.model.get_layer(layer_name_conv_2).get_weights()
            new_w_conv1, new_b_conv1, new_w_conv2 = Net2Net._wider_conv2d_weight(
                w_conv1, b_conv1, w_conv2, filter_num, "net2wider")

            new_model.model.get_layer(layer_name_conv_1).set_weights([new_w_conv1, new_b_conv1])
            # new_model.model.get_layer(layer_name_conv_2).set_weights([new_w_conv2, b_conv2])

            if i == 0:
                w_conv_group = new_w_conv2
                b_conv_group = b_conv2
            else:
                w_conv_group = np.concatenate((w_conv_group, new_w_conv2), axis=3)
                b_conv_group = np.concatenate((b_conv_group, b_conv2), axis=0)

        # find group layer's next conv layer

        w_conv2, b_conv2 = model.get_layers(layer_name, next_layer=True, type=['Conv2D', 'Conv2D_Pooling', 'Group'])[0].get_weights()

        new_w_conv_group, new_b_conv_group, new_w_conv2 = Net2Net._wider_conv2d_weight(
            w_conv_group, b_conv_group, w_conv2, new_width, "net2wider")

        for i in range(node.config['group_num']):
            layer_name_conv_2 = layer_name + '_conv2d_' + str(i) + '_2'

            if i == node.config['group_num'] - 1:
                new_model.model.get_layer(layer_name_conv_2).set_weights([new_w_conv_group[:, :, :, i * filter_num:],
                                                                          new_b_conv_group[i * filter_num:]])
            else:
                new_model.model.get_layer(layer_name_conv_2).set_weights(
                    [new_w_conv_group[:, :, :, i * filter_num: (i + 1) * filter_num],
                     new_b_conv_group[i * filter_num: (i + 1) * filter_num]])

        new_model.get_layers(layer_name, next_layer=True, type=['Conv2D', 'Conv2D_Pooling', 'Group'])[0].set_weights(
            [new_w_conv2, b_conv2])

        return new_model

    # deeper from group layer or conv layer
    def deeper_conv2d(self, model, layer_name, config, with_pooling, kernel_size=3, filters='same'):
        # construct graph
        new_graph = model.graph.copy()
        if with_pooling == False:
            type = 'Conv2D'
        else:
            type = 'Conv2D_Pooling'
        new_node = Node(type=type, name='new',
                        config={'kernel_size': kernel_size, 'filters': filters}
                        )
        new_graph.deeper(layer_name, new_node)
        # logger.debug(new_graph.to_json())

        # construct model
        new_model = MyModel(config=config, graph=new_graph)

        # inherit weight
        # in fact there is no need to get w_conv1 and b_conv1
        # what we actually need is only w_conv1's shape
        # more specifically, only kh, kw and filters are needed, num_channel is not necessary
        node_type = model.graph.get_nodes(layer_name)[0].type
        if node_type == 'Conv2D' or node_type == 'Conv2D_Pooling':
            w_conv1, b_conv1 = new_model.get_layers(layer_name)[0].get_weights()
            # tensorflow kernel format: [filter_height, filter_width, in_channels, out_channels] channels_last
            # theano kernel format:     [output_channels, input_channels, filter_rows, filter_columns] channels_first
            if K.image_data_format() == "channels_first":  # theano format
                # convert_kernel function Converts a Numpy kernel matrix from Theano format to TensorFlow format, vise versa
                w_conv1 = convert_kernel(w_conv1)
            kh, kw, num_channel, filters = w_conv1.shape
        elif node_type == 'Group':
            kh, kw, filters = 3, 3, model.graph.get_nodes(layer_name)[0].config['filters']

        w_conv2, b_conv2 = new_model.get_layers(layer_name, next_layer=True)[0].get_weights()

        new_w_conv2, new_b_conv2 = Net2Net._deeper_conv2d_weight(
            kh=kh, kw=kw, filters=filters)

        new_model.get_layers(layer_name, next_layer=True)[0].set_weights([new_w_conv2, new_b_conv2])
        self.copy_weight(model, new_model)
        return new_model

    @staticmethod
    def _deeper_conv2d_weight(kw, kh, filters):
        student_w = np.zeros((kw, kh, filters, filters))
        for i in xrange(filters):
            student_w[(kw - 1) // 2, (kh - 1) // 2, i, i] = 1.
        student_b = np.zeros(filters)
        if K.image_data_format() == "channels_first":
            student_w = convert_kernel(student_w)
        return student_w, student_b

    @staticmethod
    def rand_weight_like(weight):
        assert K.image_data_format() == "channels_last", "support channels last, but you are {}".format(
            K.image_data_format())
        kw, kh, num_channel, filters = weight.shape
        kvar = K.truncated_normal((kw, kh, num_channel, filters), 0, 0.05)
        w = K.eval(kvar)
        b = np.zeros((filters,))
        return w, b

    @staticmethod
    def _convert_weight(weight, nw_size):
        w_size = weight.shape
        assert len(w_size) == len(nw_size)
        w_ratio = [_nw / _w for _nw, _w in zip(nw_size, w_size)]
        new_weight = scipy.ndimage.zoom(weight, w_ratio)
        return new_weight

    @staticmethod
    def _wider_conv2d_weight(teacher_w1, teacher_b1, teacher_w2, new_width, init, help=None):

        if K.image_data_format() == "channels_last":
            _teacher_w1 = convert_kernel(teacher_w1)
            _teacher_w2 = convert_kernel(teacher_w2)
        else:
            _teacher_w1 = teacher_w1
            _teacher_w2 = teacher_w2

        _teacher_b1 = teacher_b1
        assert _teacher_w1.shape[3] == _teacher_w2.shape[2], (
            'successive layers from teacher model should have compatible shapes ' +
            ' all shape is {} {} {}'.format(_teacher_w1.shape, _teacher_w2.shape, _teacher_b1.shape))
        assert _teacher_w1.shape[3] == _teacher_b1.shape[0], (
            'weight and bias from same layer should have compatible shapes')
        assert new_width > _teacher_w1.shape[3], (
            'new width (filters) should be bigger than the existing one')

        n = new_width - _teacher_w1.shape[3]
        if init == 'random-pad':
            new_w1 = np.random.normal(0, 0.1, size=_teacher_w1.shape[:-1] + (n,))
            new_b1 = np.ones(n) * 0.1
            new_w2 = np.random.normal(0, 0.1, size=_teacher_w2.shape[:2] + (
                n, _teacher_w2.shape[3]))
        elif init == 'net2wider':
            index = np.random.randint(_teacher_w1.shape[3], size=n)
            factors = np.bincount(index)[index] + 1.
            new_w1 = _teacher_w1[:, :, :, index]
            new_b1 = _teacher_b1[index]
            new_w2 = _teacher_w2[:, :, index, :] / factors.reshape((1, 1, -1, 1))
        else:
            raise ValueError('Unsupported weight initializer: %s' % init)

        student_w1 = np.concatenate((_teacher_w1, new_w1), axis=3)
        if init == 'random-pad':
            student_w2 = np.concatenate((_teacher_w2, new_w2), axis=2)
        elif init == 'net2wider':
            # add small noise to break symmetry, so that student model will have
            # full capacity later
            noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
            student_w2 = np.concatenate((_teacher_w2, new_w2 + noise), axis=2)
            student_w2[:, :, index, :] = new_w2
        student_b1 = np.concatenate((_teacher_b1, new_b1), axis=0)

        if K.image_data_format() == "channels_last":
            student_w1 = convert_kernel(student_w1)
            # student_b1=convert_kernel(student_b1)
            student_w2 = convert_kernel(student_w2)

        return student_w1, student_b1, student_w2


if __name__ == "__main__":
    config = MyConfig(epochs=10, verbose=1, limit_data=True, name='before', dataset_type='cifar10')
    model_l = [["Conv2D", 'Conv2D1', {'filters': 64}],
               ["MaxPooling2D", 'maxpooling2d1', {}],
               ["Group", 'Group1', {'filters': 120, 'group_num': 2}],
               ["Conv2D_Pooling", 'Conv2D_Pooling_1', {'filters': 139}],
               ["Conv2D", 'Conv2D3', {'filters': 10}],
               ['GlobalMaxPooling2D', 'GlobalMaxPooling2D1', {}],
               ['Activation', 'Activation1', {'activation_type': 'softmax'}]]
    graph = MyGraph(model_l)
    before_model = MyModel(config, graph)
    before_model.comp_fit_eval()

    net2net = Net2Net()

    model = net2net.wider_group_conv2d(before_model, layer_name='Group1', new_width=174,
                                       config=config.copy('wider_group'))
    model.comp_fit_eval()

    ''' 
    model = net2net.deeper(before_model, config=config.copy('deeper1'))
    model = net2net.deeper_conv2d(before_model, layer_name='Conv2D2', config=config.copy('deeper1'))
    model = net2net.wider_conv2d(model, layer_name='Conv2D2', width_ratio=2, config=config.copy('wide'))
    model.comp_fit_eval()

    model = net2net.wider(model, config=config)
    model.comp_fit_eval()

    model = net2net.add_skip(model, config=config.copy('skip1'))
    model.comp_fit_eval()

    model = net2net.add_group(model, config=config)
    model.comp_fit_eval()

    model.vis()
    '''
    # from IPython import embed; embed()
