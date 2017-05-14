import keras,json
import networkx as nx
import numpy as np
import os.path as osp
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.layers import Conv2D, Input, Activation, GlobalMaxPooling2D
from keras.models import Model
from keras.backend import tensorflow_backend as ktf
from networkx.readwrite import json_graph

from Config import Config
import Init


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
    TYPES = {'Node': Node}
    def default(self, obj):
        if isinstance(obj, Node):
            key = '__%s__' % obj.__class__.__name__
            return { key: obj.__dict__}
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

    def get_nodes(self, name, next_layer=False, last_layer=False):
        name2node = {node.name: node for node in self.nodes()}
        assert name in name2node.keys(), " Name must be uniqiue"
        node = name2node[name]
        if next_layer:
            return self.successors(node)
        elif last_layer:
            return self.predecessors(node)
        else:
            return [node]

    def update(self):
        self.type2ind={}
        for node in self.nodes():
            import re
            ind=int(re.findall('\w+(\d+)$',node.name)[0])
            self.type2ind[node.type]=self.type2ind.get(node.type,[])+[ind]


    def deeper(self,name,new_node):
        node=self.get_nodes(name=name)[0]
        next_node=self.get_nodes(name=name,next_layer=True)[0]
        # TODO maybe more than 1

        # assign new node
        if new_node.name == 'new':
            self.update()
            new_name=new_node.type+str(1+max(self.type2ind[new_node.type]))
            new_node.name =new_name

        if new_node.config['filters'] == 'same':
            new_node.config['filters']=node.config['filters']

        self.remove_edge(node,next_node)
        self.add_edge(node,new_node)
        self.add_edge(new_node,next_node)


    def to_model(self, input_shape):
        graph_helper = self.copy()

        assert nx.is_directed_acyclic_graph(graph_helper)
        topo_nodes = nx.topological_sort(graph_helper)

        input_tensor = Input(shape=input_shape)

        for node in topo_nodes:
            pre_nodes = graph_helper.predecessors(node)
            suc_nodes = graph_helper.successors(node)
            # TODO Now single input; future multiple input; use len to judge
            if node.type not in ['Concatenate','Add']:
                if len(pre_nodes) == 0:
                    layer_input_tensor = input_tensor
                else:
                    assert len(pre_nodes) == 1
                    layer_input_tensor = graph_helper[pre_nodes[0]][node]['tensor']

                if node.type == 'Conv2D':
                    kernel_size = node.config.get('kernel_size', 3)
                    filters = node.config['filters']

                    layer = Conv2D(kernel_size=kernel_size, filters=filters, name=node.name)

                elif node.type == 'GlobalMaxPooling2D':
                    layer = keras.layers.GlobalMaxPooling2D(name=node.name)

                elif node.type == 'Activation':
                    activation_type = node.config['activation_type']
                    layer = Activation(activation=activation_type, name=node.name)
                layer_output_tensor = layer(layer_input_tensor)
            else:
                layer_input_tensors= [graph_helper[pre_node][node]['tensor'] for pre_node in pre_nodes]
                if node.type=='Concatenate' :
                    # handle shape
                    layer_input_tensors_shapes=[
                         ktf.int_shape(layer_input_tensor)[2:] for layer_input_tensor in layer_input_tensors
                    ]
                    layer_input_tensors_shapes=np.array(layer_input_tensors_shapes)
                    new_shape=layer_input_tensors_shapes.min(axis=0)
                    layer_input_tensors= [
                        MyGraph.my_resize(layer_input_tensor,new_shape=new_shape)
                        for layer_input_tensor in layer_input_tensors
                    ]

                    layer=keras.layers.Concatenate(axis=0)
                layer_output_tensor = layer(layer_input_tensors)

            graph_helper.add_node(node, layer=layer)

            if len(suc_nodes) == 0:
                output_tensor = layer_output_tensor
            else:
                for suc_node in suc_nodes:
                    graph_helper.add_edge(node, suc_node, tensor=layer_output_tensor)

        return Model(inputs=input_tensor, outputs=output_tensor)

    @staticmethod
    def my_resize(x,new_shape):
        import tensorflow as tf
        # original_shape = ktf.int_shape(x)
        new_shape = tf.shape(x)[2:]
        new_shape *= tf.constant(np.array(new_shape).astype('int32'))
        x = ktf.permute_dimensions(x, [0, 2, 3, 1])
        x = tf.image.resize_nearest_neighbor(x, new_shape)
        x = ktf.permute_dimensions(x, [0, 3, 1, 2])

        return  x


    def to_json(self):
        data = json_graph.node_link_data(self)
        return json.dumps(data,indent=2,cls=CustomTypeEncoder)


class MyModel(object):
    def __init__(self, config=None, graph=None, model=None):
        if config is not None:
            self.config = config
        else:
            self.config = Config()

        if model is not None:
            self.config = model.config
            self.graph = model.graph.copy()
            self.model = self.graph.to_model(self.config.input_shape)

        elif graph is not None:
            self.graph = graph
            self.model = self.graph.to_model(self.config.input_shape)

        self.lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=10,
                                            min_lr=0.5e-7)
        self.early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=10)
        self.csv_logger = CSVLogger(osp.join(Init.root_dir, 'output', 'net2net.csv'))



    def get_layers(self, name, next_layer=False, last_layer=False):
        name2layer = {layer.name: layer for layer in self.model.layers}
        def _get_layer(name):
            return name2layer[name]
        nodes=self.graph.get_nodes(name, next_layer, last_layer)
        if not isinstance(nodes,list):
            nodes=[nodes]
        return map(_get_layer,[node.name for node in nodes])


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
                       callbacks=[self.lr_reducer,self.early_stopper,self.csv_logger]
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
        print('\n-- loss and accuracy --\n')
        print(score)


if __name__ == "__main__":

    dbg = True
    if dbg:
        config = Config(epochs=0, verbose=1, limit_data=9999)
    else:
        config = Config(epochs=100, verbose=1, limit_data=1)
    model_l = [["Conv2D", 'conv1', {'filters': 16}],
               ["Conv2D", 'conv2', {'filters': 64}],
               ["Conv2D", 'conv3', {'filters': 10}],
               ['GlobalMaxPooling2D', 'gmpool1', {}],
               ['Activation', 'activation1', {'activation_type': 'softmax'}]]
    graph = MyGraph(model_l)
    teacher_model = MyModel(config, graph)

    teacher_model.comp_fit_eval()
