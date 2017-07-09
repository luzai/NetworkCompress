# Dependenct: Utils, Config
import json
from IPython import embed
import os
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
from keras import regularizers
import Utils
from Utils import vis_graph, vis_model
from Config import MyConfig
from Logger import logger

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
        for i in range(self.idx * shape[3], min(shape[2], (self.idx + 1) * shape[3])):
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
        self.depth = None

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
                config = layer[2]
                _nodes.append(Node(type, name, config))

            self.add_path(_nodes)

    def get_nodes(self, name, next_layer=False, last_layer=False, type=None):
        if type is None:
            name2node = {node.name: node for node in self.nodes()}
        else:
            name2node = {node.name: node for node in self.nodes() if node.type in type}
        assert name in name2node.keys(), " Name must be uniqiue"
        node = name2node[name]
        if next_layer:
            if type is None:
                return self.successors(node)
            else:
                poss_list, begin = [], False
                for poss in nx.topological_sort(self):
                    if poss == node:
                        begin = True
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
        for node in nx.topological_sort(self):
            if node.type in ['Conv2D', 'Group', 'Conv2D_Pooling']:
                plus = 1
            else:
                plus = 0
            if len(self.predecessors(node)) == 0:
                node.depth = 0
            else:
                pre_depth = [_node.depth for _node in self.predecessors(node)]
                pre_depth = max(pre_depth)
                node.depth = self.max_depth = pre_depth + plus

    def deeper(self, name, new_node):
        node = self.get_nodes(name=name)[0]
        next_nodes = self.get_nodes(name=name, next_layer=True)

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

        # there maybe multiple next_node, for example, next_layer is a skip layer or group layer
        for next_node in next_nodes:
            self.remove_edge(node, next_node)
            self.add_edge(node, new_node)
            self.add_edge(new_node, next_node)

    def conv_pooling_layer(self, name, kernel_size, filters, kernel_regularizer_l2):
        def f(input):
            layer = Conv2D(kernel_size=kernel_size, filters=filters, name=name, padding='same',
                           activation='relu', kernel_regularizer=regularizers.l2(kernel_regularizer_l2))(input)
            layer = keras.layers.MaxPooling2D(name=name + '_maxpooling')(layer)
            return layer

        return f

    def group_layer(self, group_num, filters, name, kernel_regularizer_l2):
        def f(input):
            if group_num == 1:
                tower = Conv2D(filters, (1, 1), name=name + '_conv2d_0_1', padding='same',
                               kernel_initializer=IdentityConv())(input)
                tower = Conv2D(filters, (3, 3), name=name + '_conv2d_0_2', padding='same',
                               kernel_initializer=IdentityConv(), activation='relu',
                               kernel_regularizer=regularizers.l2(kernel_regularizer_l2))(tower)
                return tower
            else:
                group_output = []
                for i in range(group_num):
                    filter_num = filters / group_num
                    # if filters = 201, group_num = 4, make sure last group filters num = 51
                    if i == group_num - 1:  # last group
                        filter_num = filters - i * (filters / group_num)

                    tower = Conv2D(filter_num, (1, 1), name=name + '_conv2d_' + str(i) + '_1', padding='same',
                                   kernel_initializer=GroupIdentityConv(i, group_num))(input)
                    tower = Conv2D(filter_num, (3, 3), name=name + '_conv2d_' + str(i) + '_2', padding='same',
                                   kernel_initializer=IdentityConv(), activation='relu',
                                   kernel_regularizer=regularizers.l2(kernel_regularizer_l2))(tower)
                    group_output.append(tower)

                if K.image_data_format() == 'channels_first':
                    axis = 1
                elif K.image_data_format() == 'channels_last':
                    axis = 3
                output = Concatenate(axis=axis)(group_output)

                return output

        return f

    def to_model(self, input_shape, name="default_for_op", kernel_regularizer_l2=0.01):
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
                    layer = Conv2D(kernel_size=kernel_size, filters=filters,
                                   name=node.name, padding='same',
                                   activation='relu',
                                   kernel_regularizer=regularizers.l2(kernel_regularizer_l2)
                                   )
                elif node.type == 'Conv2D_Pooling':
                    kernel_size = node.config.get('kernel_size', 3)
                    filters = node.config['filters']
                    layer = self.conv_pooling_layer(name=node.name, kernel_size=kernel_size,
                                                    filters=filters, kernel_regularizer_l2=kernel_regularizer_l2)
                elif node.type == 'Group':
                    layer = self.group_layer(name=node.name, group_num=node.config['group_num'],
                                             filters=node.config['filters'],
                                             kernel_regularizer_l2=kernel_regularizer_l2)
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
                if node.type in ['Conv2D', 'Conv2D_Pooling', 'Group']:
                    self.update(), graph_helper.update()
                    # MAX_DP, MIN_DP = .35, .01
                    # ratio_dp = - (MAX_DP - MIN_DP) / self.max_depth * node.depth + MAX_DP
                    # use fixed drop out ratio
                    ratio_dp = 0.30
                    layer_output_tensor = keras.layers.Dropout(ratio_dp)(layer_output_tensor)
                    # logger.debug('layer {} ratio of dropout {}'.format(node.name, ratio_dp))

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
                                keras.layers.MaxPool2D(pool_size=diff_shape, strides=1, name=node.name + '_conv2d_pooling')(
                                    layer_input_tensor)
                        if ori_chnls[ind] > new_chnl:
                            layer_input_tensors[ind] = \
                                Conv2D(filters=new_chnl, kernel_size=1, padding='same',
                                       name=node.name + '_conv2d')(layer_input_tensor)

                        layer = keras.layers.Add(name=node.name)
                        # logger.debug('In graph to_model add a Add layer with name {}'.format(node.name))

                if node.type == 'Concatenate':
                    logger.critical('Concatenate is decrapted!!!')
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
                    layer = keras.layers.Concatenate(axis=chn_ind, name=node.name)
                try:
                    layer_output_tensor = layer(layer_input_tensors)
                except:
                    embed()
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
            _str = json.dumps(data, indent=2, cls=CustomTypeEncoder)
        except Exception as inst:
            _str = ""
            logger.error(str(inst))
        return _str

    def save_params(self, path):
        #save depth, max width, min width, cardinality of the model
        depth = 0
        cardinality = 1
        max_width = -1
        min_width = 120
        for node in self.nodes():
            if node.type == 'Conv2D' or node.type == 'Conv2D_Pooling' or node.type == 'Group':
                depth = depth + 1
                if node.type == 'Group':
                    cardinality = cardinality * node.config['group_num']
                if node.config['filters'] > max_width:
                    max_width = node.config['filters']
            if node.type == 'Add':
                cardinality = cardinality * 2
        cardinality = cardinality * depth
        sav = {}
        sav['c'] = cardinality
        sav['h'] = depth
        sav['w'] = max_width
        sav['w0'] = min_width

        #depressed
        '''
        E = len(self.edges())
        V = len(self.nodes())
        for node in self.nodes():
            if node.type.lower() == 'group':
                E += node.config['group_num']*2
                V += node.config['group_num']
        sav = {}
        sav['E'] = E
        sav['V'] = V
        '''
        Utils.write_json(sav, path)


class MyModel(object):
    def __init__(self, config, graph=None, model=None):
        self.config = config
        self.graph = graph
        if model is None:
            self.model = self.graph.to_model(
                self.config.input_shape,
                name=self.config.name,
                kernel_regularizer_l2=self.config.kernel_regularizer_l2)
        else:
            self.model = model

    def get_layers(self, name, next_layer=False, last_layer=False, type=None):
        if type is None:
            name2layer = {layer.name: layer for layer in self.model.layers}
        else:
            name2layer = {}
            for layer in self.model.layers:
                for t in type:
                    if t.lower() in layer.name.lower():
                        name2layer[layer.name] = layer
                        break
                        # name2layer = {layer.name: layer for layer in self.model.layers if type.lower() in layer.name.lower()}

        def _get_layer(name):
            return name2layer[name]

        nodes = self.graph.get_nodes(name, next_layer, last_layer, type=type)
        if not isinstance(nodes, list):
            nodes = [nodes]
        for node in nodes:
            if node.name not in name2layer:
                embed()
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
                              callbacks=[self.config.lr_reducer,
                                         self.config.csv_logger,
                                         self.config.early_stopper,
                                         TensorBoard(log_dir=self.config.tf_log_path,
                                                     # histogram_freq=20,
                                                     # batch_size=32,
                                                     # write_graph=True,
                                                     # write_grads=True,
                                                     # write_images=True,
                                                     # embeddings_freq=0
                                                     )]
                              )
        # todo do we need earlystop?
        logger.info("Fit model {} Consume {}:".format(self.config.name, time.time() - tic))
        return hist

    def evaluate(self):
        score = self.model.evaluate(self.config.dataset['test_x'],
                                    self.config.dataset['test_y'],
                                    batch_size=self.config.batch_size, verbose=self.config.verbose)
        return score

    def vis(self):
        Utils.vis_model(self.model, self.config.name)
        if self.graph is not None:
            Utils.vis_graph(self.graph, self.config.name, show=False)
        logger.info("Vis model {} :".format(self.config.name))
        self.model.summary()
        trainable_count, non_trainable_count = Utils.count_weight(self.model)
        logger.info(
            "model {} trainable weight {} MB, non trainable_weight {} MB".format(self.config.name,
                                                                                 trainable_count,
                                                                                 non_trainable_count))
        return trainable_count + non_trainable_count


    '''
        Fit(n) = alpha * P(n) + beta * T(n) + gama * S(n) + eta * ST(n)
        
        P(n)  = val_acc(n) / val_acc(teacher)
        ST(n) = 2 * (sig(c / (h * w / w0)) - 0.5)   
            c is cardinality of model, h is depth of model, w is max width of model
    '''
    @staticmethod
    def fitness_function(info, path):

        def sigmoid(x):
            import math
            return 1.0 / (1.0 + math.exp(-x))

        #load graph's E and V
        path = path + '/graph.json'
        if os.path.isfile(path) == True:
            with open(path, 'r') as f:
                graph_info = json.load(f)

        alpha, beta, gama, eta = 1, 1, 1, 1

        #TODO: how to set these variables
        teacher_param = 5.0
        teacher_val_acc = 0.99
        teacher_test_time = 2

        norm_param = 1 - info['param'] / teacher_param
        norm_acc = info['val_acc'] / teacher_val_acc
        norm_test_time = 1 - info['test_time'] / teacher_test_time
        norm_model_struct = 2 * (sigmoid(graph_info['c'] / (graph_info['h'] * graph_info['w'] / graph_info['w0'])) - 0.5)

        final_score = alpha * norm_param + beta * norm_acc + gama * norm_test_time + eta * norm_model_struct

        return final_score

    def comp_fit_eval(self):

        self.compile()
        weight = self.vis()
        hist = self.fit()
        import time
        tic = time.time()
        score = self.evaluate()
        test_time = time.time() - tic
        logger.info('model {} loss {} and accuracy {} \n'.format(self.config.name, score[0], score[1]))
        sav_data = {}
        sav_data['param'] = weight  # count unit is MB
        sav_data['val_acc'] = score[1]
        sav_data['test_time'] = test_time
        Utils.write_json(sav_data, self.config.output_path + '/info.json')

        #final_score = MyModel.fitness_function(sav_data, self.config.output_path)
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
