import json
import multiprocessing as mp
import time

import keras
import numpy as np
from keras.layers import Input, Conv2D, GlobalMaxPooling2D, Activation
from keras.models import Model

import GAClient
import Utils
from Config import MyConfig
from Logger import logger
from Model import MyModel, MyGraph
from Net2Net import Net2Net


class GA(object):
    # def my_model2model(self):
    #     raise IOError

    def __init__(self, gl_config, nb_inv=3):
        self.gl_config = gl_config
        self.population = {}
        self.net2net = Net2Net()
        self.max_ind = 0
        self.iter = 0
        self.nb_inv = nb_inv  # 10 later
        self.queue = mp.Queue(self.nb_inv * 2)

    '''
        define original init model 
    '''

    def make_init_model(self):
        models = []

        input_data = Input(shape=self.gl_config.input_shape)
        import random
        init_model_index = random.randint(1, 4)

        if init_model_index == 1:  # one conv layer with kernel num = 64
            stem_conv_1 = Conv2D(120, 3, padding='same', name='conv2d1')(input_data)

        elif init_model_index == 2:  # two conv layers with kernel num = 64
            stem_conv_0 = Conv2D(120, 3, padding='same', name='conv2d1')(input_data)
            stem_conv_1 = Conv2D(120, 3, padding='same', name='conv2d2')(stem_conv_0)

        elif init_model_index == 3:  # one conv layer with a wider kernel num = 128
            stem_conv_1 = Conv2D(120, 3, padding='same', name='conv2d1')(input_data)

        elif init_model_index == 4:  # two conv layers with a wider kernel_num = 128
            stem_conv_0 = Conv2D(120, 3, padding='same', name='conv2d1')(input_data)
            stem_conv_1 = Conv2D(120, 3, padding='same', name='conv2d2')(stem_conv_0)
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

    def get_curr_config(self):
        return self.gl_config.copy('ga_iter_' + str(self.iter) + '_ind_' + str(self.max_ind))

    def mutation_process(self):
        if self.iter != 0:
            for name, model in self.population.items():
                model.model.load_weights(model.config.model_path)

        self.iter += 1
        for name, before_model in self.population.items():
            self.max_ind += 1
            new_config = self.get_curr_config()
            suceeded = False
            while not suceeded:
                evolution_choice_list = ['deeper', 'wider', 'add_skip', 'add_group']
                evolution_choice = np.random.choice(evolution_choice_list, 1)[0]
                logger.info("evolution choice {}".format(evolution_choice))
                try:
                    if evolution_choice == 'deeper':
                        after_model = self.net2net.deeper(before_model, config=new_config)
                        suceeded = True
                    elif evolution_choice == 'wider':
                        after_model = self.net2net.wider(before_model, config=new_config)
                        suceeded = True
                    elif evolution_choice == 'add_skip':
                        after_model = self.net2net.add_skip(before_model, config=new_config)
                        suceeded = True
                    elif evolution_choice == 'add_group':
                        after_model = self.net2net.add_group(before_model, config=new_config)
                        suceeded = True
                except Exception as inst:
                    logger.error(
                        "before model {} after model {} evoultion choice {} fail ref to detailed summary".format(
                            before_model.config.name, after_model.config.name, evolution_choice))
                    before_model.model.summary()
                    after_model.model.summary()

            assert 'after_model' in locals()
            # TODO interface for deep wide + weight copy
            setattr(after_model, 'parent', before_model.config.name)
            self.population[after_model.config.name] = after_model

    def train_process(self):
        clients = []

        for model in self.population.values():

            # if getattr(model, 'parent', None) is not None:
            # has parents means muatetion and weight change, so need to save weights
            keras.models.save_model(model.model, model.config.model_path)

            d = dict(
                queue=self.queue,
                name=model.config.name,
                epochs=model.config.epochs,
                verbose=model.config.verbose,
                limit_data=model.config.limit_data,
                dataset_type=model.config.dataset_type
            )
            if parallel:
                c = mp.Process(target=GAClient.run_self, kwargs=d)
                c.start()
                time.sleep(np.random.choice([0, 5, 10]))
                clients.append(c)
            else:
                GAClient.run(**d)
                d = self.queue.get()
                name = d[0]
                score = d[1]
                setattr(self.population[name], 'score', score)

        if parallel:
            cnt = 0
            for c in clients:
                c.join()
            while not self.queue.empty():
                d = self.queue.get()
                name = d[0]
                score = d[1]
                setattr(self.population[name], 'score', score)
                cnt += 1
            if cnt != len(clients):
                from IPython import embed
                embed()
                time.sleep(-1)

    def ga_main(self):
        for i in range(self.nb_inv):
            model = self.make_init_model()
            model_l = self.get_model_list(model)
            graph = MyGraph(model_l)
            config = self.get_curr_config()
            self.population[config.name] = MyModel(config=config, graph=graph)
            self.max_ind += 1
        for i in range(self.gl_config.evoluation_time):
            logger.info("Now {} evolution ".format(i + 1))
            if dbg:
                self.mutation_process()
                self.select_process()
                self.train_process()
            else:
                self.mutation_process()
                self.train_process()
                self.select_process()

    def select_process(self):
        # for debug, just keep the latest evolutioned model
        if dbg:
            self.population = Utils.choice_dict_keep_latest(self.population, self.nb_inv)
            return

        for model in self.population.values():
            assert hasattr(model, 'score'), 'to eval we need score'
        self.population = Utils.choice_dict(self.population, self.nb_inv)

        if len(np.unique(self.population.keys())) == self.nb_inv:
            logger.warning("!warning: sample without replacement")


if __name__ == "__main__":
    global parallel, dbg
    dbg = False
    if dbg:
        parallel = False  # if want to dbg set epochs=1 and limit_data=True
        gl_config = MyConfig(epochs=1, verbose=2, limit_data=True, name='ga', evoluation_time=10)
        nb_inv = 5
    else:
        parallel = True
        gl_config = MyConfig(epochs=50, verbose=2, limit_data=False, name='ga', evoluation_time=3, dataset_type='mnist')
        nb_inv = 3
    ga = GA(gl_config=gl_config, nb_inv=nb_inv)
    ga.ga_main()
