# Dependenct: Utils, Config
import json

import keras
import networkx as nx
import numpy as np
from keras.backend import tensorflow_backend as ktf
from keras.callbacks import TensorBoard
from keras.layers import Conv2D, Input, Activation
from keras.models import Model
from networkx.readwrite import json_graph
from keras.layers.merge import Concatenate
import keras.backend as K
from keras.utils.conv_utils import convert_kernel
from keras.initializers import Initializer
import Utils
from Utils import vis_graph, vis_model
from Config import MyConfig
from Logger import logger


class IdentityConv(Initializer):
    def __call__(self, shape, dtype=None):
        assert K.image_data_format() == 'channels_last'
        # kw,kh,num_channel,filters
        if len(shape) == 1:
            return K.tensorflow_backend.constant(0., dtype=dtype, shape=shape)
        elif len(shape) == 2 and shape[0] == shape[1]:
            return K.tensorflow_backend.constant(np.identity(shape[0], dtype))
        elif len(shape) == 4 and shape[2] == shape[3]:
            array = np.zeros(shape, dtype=float)
            cx, cy = shape[0] / 2, shape[1] / 2
            for i in range(shape[2]):
                array[cx, cy, i, i] = 1
            return K.tensorflow_backend.constant(array, dtype=dtype)
        elif len(shape) == 4 and shape[2] != shape[3]:
            array = np.zeros(shape, dtype=float)
            cx, cy = (shape[0] - 1) // 2, (shape[1] - 1) // 2
            for i in range(min(shape[2], shape[3])):
                array[cx, cy, i, i] = 1
            return K.tensorflow_backend.constant(array, dtype=dtype)
        else:
            raise Exception("no handler")


class GroupIdentityConv(Initializer):
    def __init__(self, idx, group_num):
        self.idx = idx
        self.group_num = group_num

    def __call__(self, shape, dtype=None):
        assert K.image_data_format() == 'channels_last'
        # kw,kh,num_channel,filters

        array = np.zeros(shape, dtype=float)
        cx, cy = (shape[0] - 1) // 2, (shape[1] - 1) // 2
        cnt = 0
        for i in range(self.idx * shape[3], min(shape[2],(self.idx + 1) * shape[3])):
            array[cx, cy, i, cnt] = 1
            cnt = cnt + 1
        return K.tensorflow_backend.constant(array, dtype=dtype)

    def get_config(self):
        return {
            'idx': self.idx,
            'group_num': self.group_num
        }


class Node(object):
    def __init__(self, type, name, config):
        self.type = type
        self.name = name
        self.config = config
        # decrypted:
        # self.input_tensors = []
        # self.output_tensors = []

    def __str__(self):
        return self.name


class CustomTypeEncoder(json.JSONEncoder):
    """A custom JSONEncoder class that knows how to encode core custom
    objects.

    Custom objects are encoded as JSON object literals (ie, dicts) with
    one key, '__TypeName__' where 'TypeName' is the actual name of the
    type to which the object belongs.  That single key maps to another
    object literal which is just the __dict__ of the object encoded."""

    # TYPES = {'Node': Node}
    def default(self, obj):
        if isinstance(obj, Node) or isinstance(obj, keras.layers.Layer):
            key = '__%s__' % obj.__class__.__name__
            return {key: obj.__dict__}
        return json.JSONEncoder.default(self, obj)


class MyGraph(nx.DiGraph):
    def __init__(self, model_l=None):
        super(MyGraph, self).__init__()
        if model_l is not None:
            _nodes = []
            for layer in model_l:
                type = layer[0]
                name = layer[1]
                # name_ind = int(re.findall(r'\d+', name)[0])
                config = layer[2]
                _nodes.append(Node(type, name, config))
                # if type not in self.type2inds.keys():
                #     self.type2inds[type] = [name_ind]
                # else:
                #     self.type2inds[type] += [name_ind]

            self.add_path(_nodes)

    def get_nodes(self, name, next_layer=False, last_layer=False, type=None):
        if type is None:
            name2node = {node.name: node for node in self.nodes()}
        else:
            name2node = {node.name: node for node in self.nodes() if node.type == type}
        assert name in name2node.keys(), " Name must be uniqiue"
        node = name2node[name]
        if next_layer:
            if type is None:
                return self.successors(node)
            else:
                poss_list,begin=[],False
                for poss in nx.topological_sort(self):
                    if poss == node:
                        begin=True
                        continue
                    if begin and poss in name2node.values():
                        poss_list.append(poss)
                return [poss_list[0]]
        elif last_layer:
            return self.predecessors(node)
        else:
            return [node]

    def update(self):

        self.type2ind = {}
        for node in self.nodes():
            import re
            ind = int(re.findall(r'^\w+?(\d+)$', node.name)[0])
            self.type2ind[node.type] = self.type2ind.get(node.type, []) + [ind]

    def deeper(self, name, new_node):
        node = self.get_nodes(name=name)[0]
        next_node = self.get_nodes(name=name, next_layer=True)[0]
        # TODO maybe more than 1

        # assign new node
        if new_node.name == 'new':
            self.update()
            new_name = new_node.type + \
                       str(
                           1 + max(self.type2ind.get(new_node.type, [0]))
                       )
            new_node.name = new_name

        if new_node.config['filters'] == 'same':
            new_node.config['filters'] = node.config['filters']

        self.remove_edge(node, next_node)
        self.add_edge(node, new_node)
        self.add_edge(new_node, next_node)

    def group_layer(self, input_tensor, group_num, filters, name):
        def f(input):
            if group_num == 1:
                tower = Conv2D(filters, (1, 1), name=name + '_conv2d_0_1', padding='same',
                               kernel_initializer=IdentityConv())(input)
                tower = Conv2D(filters, (3, 3), name=name + '_conv2d_0_2', padding='same',
                               kernel_initializer=IdentityConv(), activation='relu')(tower)
                return tower
            else:
                group_output = []
                for i in range(group_num):
                    filter_num = filters / group_num
                    #if filters = 201, group_num = 4, make sure last group filters num = 51
                    if i == group_num - 1: # last group
                        filter_num = filters - i * (filters / group_num)

                    tower = Conv2D(filter_num, (1, 1), name=name + '_conv2d_' + str(i) + '_1', padding='same',
                                   kernel_initializer=GroupIdentityConv(i, group_num))(input)
                    tower = Conv2D(filter_num, (3, 3), name=name + '_conv2d_' + str(i) + '_2', padding='same',
                                   kernel_initializer=IdentityConv(), activation='relu')(tower)
                    group_output.append(tower)

                if K.image_data_format() == 'channels_first':
                    axis = 1
                elif K.image_data_format() == 'channels_last':
                    axis = 3
                output = Concatenate(axis=axis)(group_output)

                return output

        return f

    def to_model(self, input_shape, graph, name="default_for_op"):
        # with graph.as_default():
        #     with tf.name_scope(name) as scope:
        graph_helper = self.copy()

        assert nx.is_directed_acyclic_graph(graph_helper)
        topo_nodes = nx.topological_sort(graph_helper)

        input_tensor = Input(shape=input_shape)

        for node in topo_nodes:
            pre_nodes = graph_helper.predecessors(node)
            suc_nodes = graph_helper.successors(node)

            if node.type not in ['Concatenate', 'Add', 'Multiply']:
                if len(pre_nodes) == 0:
                    layer_input_tensor = input_tensor
                else:
                    assert len(pre_nodes) == 1
                    layer_input_tensor = graph_helper[pre_nodes[0]][node]['tensor']

                if node.type == 'Conv2D':
                    kernel_size = node.config.get('kernel_size', 3)
                    filters = node.config['filters']

                    layer = Conv2D(kernel_size=kernel_size, filters=filters, name=node.name, padding='same',
                                   activation='relu')
                elif node.type == 'Group':
                    layer = self.group_layer(layer_input_tensor, name=node.name, group_num=node.config['group_num'],
                                             filters=node.config['filters'])

                elif node.type == 'GlobalMaxPooling2D':
                    layer = keras.layers.GlobalMaxPooling2D(name=node.name)
                elif node.type == 'MaxPooling2D':
                    layer = keras.layers.MaxPooling2D(name=node.name)
                elif node.type == 'AveragePooling2D':
                    layer = keras.layers.AveragePooling2D(name=node.name)
                elif node.type == 'Activation':
                    activation_type = node.config['activation_type']
                    layer = Activation(activation=activation_type, name=node.name)
                layer_output_tensor = layer(layer_input_tensor)
            else:
                layer_input_tensors = [graph_helper[pre_node][node]['tensor'] for pre_node in pre_nodes]
                if node.type == 'Add':
                    # todo also test multiply
                    assert K.image_data_format() == 'channels_last'
                    ori_shapes = [ktf.int_shape(layer_input_tensor)[1:3]
                                  for layer_input_tensor in layer_input_tensors]
                    ori_shapes = np.array(ori_shapes)
                    new_shape = ori_shapes.min(axis=0)
                    ori_chnls = [ktf.int_shape(layer_input_tensor)[3]
                                 for layer_input_tensor in layer_input_tensors]
                    ori_chnls = np.array(ori_chnls)
                    new_chnl = ori_chnls.min()

                    for ind, layer_input_tensor, ori_shape in \
                            zip(range(len(layer_input_tensors)), layer_input_tensors, ori_shapes):
                        diff_shape = ori_shape - new_shape
                        if diff_shape.any():
                            diff_shape += 1
                            layer_input_tensors[ind] = \
                                keras.layers.MaxPool2D(pool_size=diff_shape, strides=1, name=node.name + '_maxpool2d')(
                                    layer_input_tensor)
                        if ori_chnls[ind] > new_chnl:
                            layer_input_tensors[ind] = \
                                Conv2D(filters=new_chnl, kernel_size=1, padding='same',
                                       name=node.name + '_conv2d')(layer_input_tensor)
                        layer = keras.layers.Add()

                if node.type == 'Concatenate':
                    if K.image_data_format() == "channels_last":
                        (width_ind, height_ind, chn_ind) = (1, 2, 3)
                    else:
                        (width_ind, height_ind, chn_ind) = (2, 3, 1)
                    ori_shapes = [
                        ktf.int_shape(layer_input_tensor)[width_ind:height_ind + 1] for layer_input_tensor in
                        layer_input_tensors
                    ]
                    ori_shapes = np.array(ori_shapes)
                    new_shape = ori_shapes.min(axis=0)
                    for ind, layer_input_tensor, ori_shape in \
                            zip(range(len(layer_input_tensors)), layer_input_tensors, ori_shapes):
                        diff_shape = ori_shape - new_shape
                        if diff_shape.all():
                            diff_shape += 1
                            layer_input_tensors[ind] = \
                                keras.layers.MaxPool2D(pool_size=diff_shape, strides=1)(layer_input_tensor)
                    # todo custom div layer
                    # def div2(x):
                    #     return x / 2.
                    # layer_input_tensors = [keras.layers.Lambda(div2)(tensor) for tensor in layer_input_tensors]
                    layer = keras.layers.Concatenate(axis=chn_ind)
                layer_output_tensor = layer(layer_input_tensors)

            graph_helper.add_node(node, layer=layer)

            if len(suc_nodes) == 0:
                output_tensor = layer_output_tensor
            else:
                for suc_node in suc_nodes:
                    graph_helper.add_edge(node, suc_node, tensor=layer_output_tensor)
        # assert tf.get_default_graph() == graph, "should be same"
        # tf.train.export_meta_graph('tmp.pbtxt', graph_def=tf.get_default_graph().as_graph_def())
        assert 'output_tensor' in locals()
        import time
        tic = time.time()
        model = Model(inputs=input_tensor, outputs=output_tensor)
        logger.info('Consume Time(Just Build model: {}'.format(time.time() - tic))

        return model

    def to_json(self):
        data = json_graph.node_link_data(self)
        try:
            str = json.dumps(data, indent=2, cls=CustomTypeEncoder)
        except Exception as inst:
            str = ""
            print inst

        return str


class MyModel(object):
    def __init__(self, config, graph=None, model=None):
        self.config = config
        if model is None:
            self.graph = graph
            self.model = self.graph.to_model(
                self.config.input_shape,
                # graph=self.config.tf_graph,
                graph=None,
                name=self.config.name)
        else:
            self.model = model

    def get_layers(self, name, next_layer=False, last_layer=False, type=None):
        if type is None:
            name2layer = {layer.name: layer for layer in self.model.layers}
        else:
            name2layer = {layer.name: layer for layer in self.model.layers if type.lower() in layer.name.lower()}

        def _get_layer(name):
            return name2layer[name]

        nodes = self.graph.get_nodes(name, next_layer, last_layer, type=type)
        if not isinstance(nodes, list):
            nodes = [nodes]
        return map(_get_layer, [node.name for node in nodes])

    def compile(self):
        self.model.compile(optimizer='adam',  # rmsprop
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self):
        import time
        tic = time.time()
        logger.info("Start train model {}\n".format(self.config.name))
        hist = self.model.fit(self.config.dataset['train_x'],
                              self.config.dataset['train_y'],
                              # validation_split=0.2,
                              validation_data=(self.config.dataset['test_x'], self.config.dataset['test_y']),
                              verbose=self.config.verbose,
                              batch_size=self.config.batch_size,
                              epochs=self.config.epochs,
                              callbacks=[self.config.lr_reducer, self.config.csv_logger, self.config.early_stopper,
                                         TensorBoard(log_dir=self.config.tf_log_path)]
                              )
        # todo earlystop?
        logger.info("Fit model {} Consume {}:".format(self.config.name, time.time() - tic))
        return hist

    def evaluate(self):
        score = self.model.evaluate(self.config.dataset['test_x'],
                                    self.config.dataset['test_y'],
                                    batch_size=self.config.batch_size, verbose=self.config.verbose)
        return score

    def vis(self):
        Utils.vis_model(self.model, self.config.name)
        if hasattr(self, 'graph'):
            Utils.vis_graph(self.graph, self.config.name, show=False)
        logger.info("Vis model {} :".format(self.config.name))
        self.model.summary()
        trainable_count, non_trainable_count = Utils.count_weight(self.model)
        logger.info(
            "model {} trainable weight {} MB, non trainable_weight {} MB".format(self.config.name,
                                                                                 trainable_count,
                                                                                 non_trainable_count))

    def comp_fit_eval(self):

        self.compile()
        self.vis()
        hist = self.fit()

        score = self.evaluate()
        logger.info('model {} loss {} and accuracy {} \n'.format(self.config.name, score[0], score[1]))

        return score[-1]


if __name__ == "__main__":
    dbg = True
    if dbg:
        config = MyConfig(epochs=1, verbose=1, limit_data=dbg, name='model_test')
    else:
        config = MyConfig(epochs=100, verbose=1, limit_data=dbg, name='model_test')
    model_l = [["Conv2D", 'conv1', {'filters': 16}],
               ["Group", 'group1', {'group_num': 4, 'filters': 16}],
               ["Conv2D", 'conv3', {'filters': 10}],
               ['GlobalMaxPooling2D', 'gmpool1', {}],
               ['Activation', 'activation1', {'activation_type': 'softmax'}]]
    graph = MyGraph(model_l)
    teacher_model = MyModel(config, graph)
    teacher_model.comp_fit_eval()
