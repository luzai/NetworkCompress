import keras,tensorflow
import json
import numpy as np
from keras.layers import Input, Conv2D, GlobalMaxPooling2D, Activation
from keras.models import Model
import multiprocessing as mp
import Config
import Utils
from Config import MyConfig
from Model import MyModel, MyGraph
from Net2Net import Net2Net
import GAClient


class GA(object):
    # def my_model2model(self):
    #     raise IOError

    def __init__(self, gl_config):
        self.gl_config = gl_config
        self.population = {}
        self.net2net = Net2Net()
        self.max_ind = 0
        self.iter = 0
        self.nb_inv = 3
        self.queue = mp.Queue(self.nb_inv)

    '''
        define original init model 
    '''

    def make_init_model(self):

        input_data = Input(shape=self.gl_config.input_shape)
        import random
        init_model_index = random.randint(1, 4)

        if init_model_index == 1:  # one conv layer with kernel num = 64
            stem_conv_1 = Conv2D(64, 3, padding='same', name='conv2d1')(input_data)

        elif init_model_index == 2:  # two conv layers with kernel num = 64
            stem_conv_0 = Conv2D(64, 3, padding='same', name='conv2d1')(input_data)
            stem_conv_1 = Conv2D(64, 3, padding='same', name='conv2d2')(stem_conv_0)

        elif init_model_index == 3:  # one conv layer with a wider kernel num = 128
            stem_conv_1 = Conv2D(128, 3, padding='same', name='conv2d2')(input_data)

        elif init_model_index == 4:  # two conv layers with a wider kernel_num = 128
            stem_conv_0 = Conv2D(128, 3, padding='same', name='conv2d1')(input_data)
            stem_conv_1 = Conv2D(128, 3, padding='same', name='conv2d2')(stem_conv_0)
        import keras
        stem_conv_1 = keras.layers.MaxPooling2D(name='maxpooling2d1')(stem_conv_1)
        stem_conv_1 = Conv2D(10, 3, padding='same', name='conv2d3')(stem_conv_1)
        stem_global_pooling_1 = GlobalMaxPooling2D(name='globalmaxpooling2d1')(stem_conv_1)
        stem_softmax_1 = Activation('softmax', name='activation1')(stem_global_pooling_1)

        model = Model(inputs=input_data, outputs=stem_softmax_1)
        return model

    def get_model_list(self, model):
        model_list = []
        model_dict = json.loads(model.to_json())

        model_layer = model_dict['config']['layers']

        for layer in model_layer:
            layer_name = layer['config']['name']
            layer_output_shape = model.get_layer(layer_name).output_shape
            if layer['class_name'] == 'Conv2D' and layer['config']['name'].lower().startswith('conv'):
                model_list.append([layer['class_name'], layer['config']['name'],
                                   {'kernel_size': layer['config']['kernel_size'],
                                    'filters': layer['config']['filters']}])
            elif layer['class_name'] == 'GlobalMaxPooling2D':
                model_list.append([layer['class_name'],
                                   layer['config']['name'],
                                   {}])
            elif layer['class_name'] == 'Activation':
                model_list.append([layer['class_name'],
                                   layer['config']['name'],
                                   {'activation_type': 'softmax'}])

        return model_list

    def evolution_process(self):
        self.iter += 1
        for name,before_model in self.population.items():
            self.max_ind += 1
            new_config = self.gl_config.copy('ga_iter_' + str(self.iter) + '_' + str(self.max_ind))

            evolution_choice_list = ['deeper']  # , 'wider','add_skip']
            evolution_choice = np.random.choice(evolution_choice_list, 1)[0]
            print evolution_choice

            if evolution_choice == 'deeper':
                after_model = self.net2net.deeper(before_model, config=new_config)
            elif evolution_choice == 'wider':
                after_model = self.net2net.wider(before_model, config=new_config)
            elif evolution_choice == 'add_skip':
                after_model = self.net2net.add_skip(before_model, config=new_config)
            assert 'after_model' in locals()
            # TODO interface for deep wide + weight copy
            setattr(after_model, 'parent', before_model.config.name)
            self.population[after_model.config.name] = after_model

    def fit_model_process(self):
        clients = []
        for model in self.population.values():
            # if getattr(model, 'parent', None) is not None:
                # has parents means muatetion and weight change, so need to save weights
            keras.models.save_model(model.model, model.config.model_path)


            d = dict(
                queue=self.queue,
                name=model.config.name,
                epochs=model.config.epochs,
                verbose=model.config.verbose
            )
            c = mp.Process(target=GAClient.run, kwargs=d)
            c.start()
            clients.append(c)
        for i in range(len(clients)):
            d = self.queue.get()
            name = d[0]
            score = d[1]
            setattr( self.population[name],'score',score)

        for c in clients:
            c.join()

    def genetic_grow_model(self):
        Utils.mkdir_p('output/ga/')
        model_l = self.get_model_list(self.make_init_model())
        graph = MyGraph(model_l)
        self.population[self.gl_config.name] = MyModel(config=self.gl_config, graph=graph)
        for i in range(self.gl_config.evoluation_time):
            Config.logger.info("Now {} individual ".format(i))
            self.evolution_process()
            self.fit_model_process()
            self.select_process()

    def select_process(self):
        all_name=self.population.keys()
        choose_ind=np.random.permutation(len(self.population))[:self.nb_inv]
        choose_name=all_name[choose_ind]
        self.population={name:model for name ,model in self.population.items() if name in choose_name}

        assert len(np.unique(self.population)) == self.nb_inv, 'individual should not be same'
        for name,model in self.population:
            model.model.load_weight(model.config.model_path)

if __name__ == "__main__":
    dbg = True
    if dbg:
        gl_config = MyConfig(epochs=1, verbose=2, dbg=dbg, name='ga', evoluation_time=10)
    else:
        gl_config = MyConfig(epochs=100, verbose=1, dbg=dbg, name='ga', evoluation_time=100)

    ga = GA(gl_config=gl_config)
    ga.genetic_grow_model()
