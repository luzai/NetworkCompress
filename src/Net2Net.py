# !/usr/bin/env python
# Dependence: Model,Config
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy
import scipy.ndimage

import keras.backend as K
from keras.utils.conv_utils import convert_kernel
from Config import MyConfig, logger
from Model import MyModel, MyGraph, Node


class Net2Net(object):
    @staticmethod
    def my_model2model(model):
        if isinstance(model, MyModel):
            return model.model
        else:
            return model

    def deeper(self,model,config):
        # select
        names = [node.name
                     for node in model.graph.nodes()
                     if node.type=='Conv2D']

        while True:
            import random
            choice=names[np.random.randint(0,len(names))]
            next_nodes= model.graph.get_nodes(choice,next_layer=True,last_layer=False)
            if 'GlobalMaxPooling2D' not in [ node.type for node in next_nodes]:
                break

        # grow
        return self.deeper_conv2d(model,choice,kernel_size=3,filters='same',config=config)


    def wider(self,model,config):
        pass

    def add_skip(self,model,config):
        # TODO selct by using topology sort
        pass



    def copy_weight(self, before_model, after_model):

        _before_model = self.my_model2model(model=before_model)
        _after_model = self.my_model2model(model=after_model)

        layer_names = [l.name for l in _before_model.layers]
        for name in layer_names:
            weights = _before_model.get_layer(name=name).get_weights()
            try:
                _after_model.get_layer(name=name).set_weights(weights)
            except Exception as inst:
                logger.warning("except {}".format(inst))

    def skip(self, model, from_name, to_name, config=MyConfig()):

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
        new_model = MyModel(config=config, graph=new_graph)
        self.copy_weight(model, new_model)

        return new_model

    def wider_conv2d(self, model, layer_name, width_ratio, config=MyConfig()):
        # modify graph
        new_graph = model.graph.copy()
        new_node = new_graph.get_nodes(layer_name)[0]
        assert new_node.type == 'Conv2D', 'must wide a conv'
        new_width = new_node.config['filters'] * width_ratio
        new_node.config['filters'] = new_width

        logger.debug(new_graph.to_json())
        # construct model
        new_model = MyModel(config=config, graph=new_graph)

        # inherit weight
        w_conv1, b_conv1 = model.get_layers(layer_name)[0].get_weights()
        w_conv2, b_conv2 = model.get_layers(layer_name, next_layer=True)[0].get_weights()

        new_w_conv1, new_b_conv1, new_w_conv2 = Net2Net._wider_conv2d_weight(
            w_conv1, b_conv1, w_conv2, new_width, "net2wider")

        new_model.get_layers(layer_name)[0].set_weights([new_w_conv1, new_b_conv1])
        new_model.get_layers(layer_name, next_layer=True)[0].set_weights([new_w_conv2, b_conv2])
        self.copy_weight(model, new_model)
        return new_model

    def deeper_conv2d(self, model, layer_name, kernel_size=3, filters='same', config=MyConfig()):
        # construct graph
        new_graph = model.graph.copy()

        new_node = Node(type='Conv2D', name='new',
                        config={'kernel_size': kernel_size, 'filters': filters}
                        )
        new_graph.deeper(layer_name, new_node)
        logger.debug(new_graph.to_json())

        # construc model
        new_model = MyModel(config=config, graph=new_graph)

        # inherit weight
        w_conv1, b_conv1 = new_model.get_layers(layer_name)[0].get_weights()
        w_conv2, b_conv2 = new_model.get_layers(layer_name, next_layer=True)[0].get_weights()

        new_w_conv2, new_b_conv2 = Net2Net._deeper_conv2d_weight(
            w_conv1)

        new_model.get_layers(layer_name, next_layer=True)[0].set_weights([new_w_conv2, new_b_conv2])
        self.copy_weight(model, new_model)
        return new_model

    @staticmethod
    def _deeper_conv2d_weight(teacher_w1):
        if K.image_data_format()=="Channels_last":
            teacher_w1=convert_kernel(teacher_w1)
        kw, kh, num_channel, filters = teacher_w1.shape
        student_w = np.zeros((kw, kh, filters, filters))
        for i in xrange(filters):
            student_w[(kw - 1) // 2, (kh - 1) // 2, i, i] = 1.
        student_b = np.zeros(filters)
        if K.image_data_format()=="Channels_last":
            student_w=convert_kernel(student_w)
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

        if K.image_data_format()=="channels_last":
            _teacher_w1=convert_kernel(teacher_w1)

            _teacher_w2=convert_kernel(teacher_w2)
        else:
            _teacher_w1 = teacher_w1

            _teacher_w2 = teacher_w2
        _teacher_b1 = teacher_b1
        assert _teacher_w1.shape[3] == _teacher_w2.shape[2], (
            'successive layers from teacher model should have compatible shapes')
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

        if K.image_data_format()=="channels_last":
            student_w1=convert_kernel(student_w1)
            # student_b1=convert_kernel(student_b1)
            student_w2=convert_kernel(student_w2)

        return student_w1, student_b1, student_w2


if __name__ == "__main__":

    dbg = False
    if dbg:
        config = MyConfig(epochs=0, verbose=2, dbg=dbg, name='before')
    else:
        config = MyConfig(epochs=100, verbose=1, dbg=dbg, name='before')
    model_l = [["Conv2D", 'Conv2D1', {'filters': 16}],
               ["Conv2D", 'Conv2D2', {'filters': 64}],
               ["Conv2D", 'Conv2D3', {'filters': 10}],
               ['GlobalMaxPooling2D', 'GlobalMaxPooling2D1', {}],
               ['Activation', 'Activation1', {'activation_type': 'softmax'}]]
    graph = MyGraph(model_l)
    before_model = MyModel(config, graph)
    before_model.comp_fit_eval()

    net2net = Net2Net()
    after_model = net2net.wider_conv2d(before_model, layer_name='Conv2D2', width_ratio=2,config=config.copy('wide'))
    after_model.comp_fit_eval()
    after_model = net2net.deeper_conv2d(after_model, layer_name='Conv2D2',config=config.copy('depper'))
    after_model.comp_fit_eval()
    after_model = net2net.skip(after_model, from_name='Conv2D1', to_name='Conv2D2', config=config.copy('skip'))
    after_model.comp_fit_eval()

    # from IPython import embed; embed()
