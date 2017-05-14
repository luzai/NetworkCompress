from Log import logger
from init import root_dir
from Config import  Config

import keras
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout, Activation, BatchNormalization, Embedding
from keras.models import Sequential, model_from_json, Model
from keras.optimizers import SGD
from keras.utils import np_utils, vis_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import os, sys, re
import NetworkX as nx
import numpy as np
import os.path as osp


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


class MyModel(object):
    def __init__(self,config=None, model_l=None,graph=None,model=None):
        if config is not None:
            self.config = config
        else:
            self.config=Config()
        if model_l is not None:
            # important : self.graph, self.model, self.type2inds
            self._list2graph(model_l)
            self._graph2model()
        elif model is not None:
            self.config=model.config
            self.graph=model.graph.copy()
            self._graph2model()
            # self.model=None
            # self.type2ind2={}
        elif config is not None and graph is not None:
            self.graph=graph.copy()
            self._graph2model()

        self.lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10,
                                            min_lr=0.5e-7)
        self.early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
        self.csv_logger = CSVLogger(osp.join(root_dir, 'output', 'net2net.csv'))

    def update(self):
        # TODO update type2inds
        pass

    def _list2graph(self, model_l):

        # will creat self.type2inds self.graph
        # decrypted
        # _nodes = [Node('Input', 'input1', {})]
        # self.type2inds = {'Input': 1}
        _nodes=[]
        self.type2inds={}
        for layer in model_l:
            type = layer[0]
            name = layer[1]
            name_ind = int(re.findall(r'\d+', name)[0])
            config = layer[2]
            _nodes.append(Node(type, name, config))
            if type not in self.type2inds.keys():
                self.type2inds[type] = [name_ind]
            else:
                self.type2inds[type] += [name_ind]

        self.graph = nx.DiGraph()
        # Decrypted
        # for ind,node in enumerate(self.nodes[:-1]):
        #     next_node=self.nodes[ind+1]
        #     self.graph.add_edge(node,next_node)

        self.graph.add_path(_nodes)

    def _graph2model(self):
        # will create self.model
        graph_helper=self.graph.copy()
        assert graph_helper.is_directed_acyclic_graph()
        topo_nodes = nx.topological_sort(graph_helper)

        input_tensor=Input(shape=self.config.input_shape)

        for node in topo_nodes:
            pre_nodes = graph_helper.predecessors(node)
            suc_nodes = graph_helper.successors(node)
            # TODO Now single input; future multiple input; use len to judge
            if len(pre_nodes) == 0:
                layer_input_tensor = input_tensor
            else:
                assert len(pre_nodes) == 1
                layer_input_tensor = graph_helper.edges[pre_nodes[0]][node]['tensor']

            if node.type == 'Conv2D':
                kernel_size = node.config.get('kernel_size', 3)
                filters = node.config['filters']

                layer=Conv2D(kernel_size, filters, name=node.name)

            elif node.type == 'GlobalMaxPooling2D':
                layer = keras.layers.GlobalMaxPooling2D(name=node.name)

            elif node.type == 'Activation':
                activation_type = node.config['activation_type']
                layer=Activation(activation_type)(layer_input_tensor)

            graph_helper.add_node(node, layer=layer)
            layer_output_tensor = layer(layer_input_tensor)
            if len(suc_nodes)==0:
                output_tensor=layer_input_tensor
            else:
                for suc_node in suc_nodes:
                    graph_helper.add_edge(node, suc_node, tensor=layer_output_tensor)

        self.model = Model(inputs=input_tensor, outputs=output_tensor)

    # decrypted
    # def get_node(self, name, next_layer=False, last_layer=False):
    #     name2node={ node.name :node for  node in  self.graph.nodes()}
    #     node=name2node[name]
    #     if next_layer:
    #         return self.graph.successors(node)
    #     elif last_layer:
    #         return self.graph.predecessors(node)
    #     else:
    #         return node

    def get_layer(self):


    def compile(self):
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def fit(self):
        self.model.fit(self.config.dataset['train_x'],
                       self.config.dataset['train_y'],
                       # validation_split=0.2,
                       validation_data=(self.config.dataset['test_x'], self.config.dataset['test_y']),
                       batch_size=self.config.batch_size,
                       epochs=self.config.epochs,
                       # callbacks=[self.lr_reducer,self.early_stopper,self.csv_logger]
                       )

    def evaluate(self):
        score = self.model.evaluate(self.config.dataset['test_x'],
                                    self.config.dataset['test_y'],
                                    batch_size=self.config.batch_size)
        return score

    def comp_fit_eval(self):
        self.compile()
        self.fit()
        score = self.evaluate()
        print('\n-- score --\n')
        print(score)

