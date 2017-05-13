from mLog import logger
from init import  root_dir

import keras
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout, Activation, BatchNormalization, Embedding
from keras.models import Sequential, model_from_json, Model
from keras.optimizers import SGD
from keras.utils import np_utils, vis_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import os,sys,re
import NetworkX as nx
import  numpy as np
import os.path as osp

class Node(object):
    def __init__(self,type,name,config):
        self.type=type
        self.name=name
        self.config=config
        # for convenience:
        self.input_tensors=[]
        self.output_tensors=[]

    def __str__(self):
        return self.name


class MyModel(object):
    def __init__(self, config,model_l):
        self.config = config
        self.lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10,min_lr=0.5e-7)
        self.early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
        self.csv_logger = CSVLogger(osp.join(root_dir, 'output', 'net2net.csv'))

        # important : self.graph, self.model, self.type2inds
        self.graph = self._list2graph(model_l)
        self.model=self._graph2model()

    def _list2graph(self,model_l):

        # will creat self.nodes self.type2inds self.graph
        self.nodes=[Node('Input','input1',{})]
        self.type2inds={'Input':1}
        for layer in model_l:
            type=layer[0]
            name=layer[1]
            name_ind=int(re.findall(r'\d+',name)[0])
            config=layer[2]
            self.nodes.append(Node(type,name,config))
            if type not in self.type2inds.keys() :
                self.type2inds[type]=[name_ind]
            else:
                self.type2inds[type]+=[name_ind]

        self.graph=nx.DiGraph()
        # Decrypted
        # for ind,node in enumerate(self.nodes[:-1]):
        #     next_node=self.nodes[ind+1]
        #     self.graph.add_edge(node,next_node)

        self.graph.add_path(self.nodes)

    def _graph2model(self):
        # will create self.model
        assert self.graph.is_directed_acyclic_graph()
        topo_nodes=nx.topological_sort(self.graph)

        for node in topo_nodes:
            pre_nodes = self.graph.predecessors(node)
            suc_nodes = self.graph.successors(node)
            if node.type == 'Input':
                x=Input(shape=self.config.input_shape)
                node.output_tensors.append(x)
                input_tensor=x
            elif node.type=='Conv2D':

                kernel_size=node.config.get('kernel_size',3)
                filters=node.config['filters']
                suc_nodes[0].input_tensors.append(
                    Conv2D(kernel_size, filters, name=node.name)(pre_nodes[0].output_tensors))
            elif node.type=='GlobalMaxPooling2D':

                suc_nodes[0].input_tensors.append(
                    keras.layers.GlobalMaxPooling2D()(pre_nodes[0].output_tensors))
            elif node.type=='Activation':
                activation_type=node.config['activation_type']
                suc_nodes[0].input_tensors.append(
                    Activation(activation_type)(pre_nodes[0].output_tensors))
        assert  input_tensor in locals()
        output_tensor=topo_nodes[-1].output_tensors[0]
        self.model = Model(inputs=input_tensor, outputs=output_tensor)


    def get_layer(self, name, next_layer=False,last_layer=False):

        if next_layer:
            ind = self.names2ind[name] + 1
        elif last_layer:
            ind = self.names2ind[name]-1
        else:
            ind = self.names2ind[name]

        return self.model.layers[ind]

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
