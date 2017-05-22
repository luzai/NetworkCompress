
import keras
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Input, Dropout, Activation, BatchNormalization, Embedding,GlobalAveragePooling2D,GlobalMaxPool2D,GlobalMaxPooling2D
from keras.models import Sequential, model_from_json, Model
from keras.optimizers import SGD
from keras.utils import np_utils, vis_utils
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K

import Config,Utils
from Net2Net import Net2Net
from Model import MyModel,MyGraph
import json
import random
from Config import MyConfig
from keras.layers import add, Input, Conv2D, MaxPooling2D, concatenate, GlobalMaxPooling2D
from keras.layers.core import Activation
from keras.models import Model

class GA(object):


    def __init__(self):
        self.population=[]
        self.net2net = Net2Net()
        self.max_ind=0

    '''
        define original init model 
    '''

    def make_init_model(self):
        input_data = Input(shape=(32, 32, 3))

        init_model_index = random.randint(1, 4)
        init_model_index = 2

        if init_model_index == 1:  # one conv layer with kernel num = 64
            stem_conv_1 = Conv2D(64, (1, 1), padding='same')(input_data)

        elif init_model_index == 2:  # two conv layers with kernel num = 64
            stem_conv_1 = Conv2D(64, (1, 1), padding='same')(input_data)
            stem_conv_2 = Conv2D(64, (1, 1), padding='same')(stem_conv_1)

        elif init_model_index == 3:  # one conv layer with a wider kernel num = 128
            stem_conv_1 = Conv2D(128, (1, 1), padding='same')(input_data)

        elif init_model_index == 4:  # two conv layers with a wider kernel_num = 128
            stem_conv_1 = Conv2D(128, (1, 1), padding='same')(input_data)
            stem_conv_2 = Conv2D(128, (1, 1), padding='same')(stem_conv_1)

        stem_global_pooling_1 = GlobalMaxPooling2D()(stem_conv_1)
        stem_softmax_1 = Activation('softmax')(stem_global_pooling_1)

        model = Model(inputs=input_data, outputs=stem_softmax_1)
        return model

    def get_model_list(self,model):
        model_list = []
        model_dict = json.loads(model.to_json())

        model_layer = model_dict['config']['layers']

        for layer in model_layer:
            layer_name = layer['config']['name']
            layer_output_shape = model.get_layer(layer_name).output_shape
            if layer['class_name'] == 'Conv2D' and layer['config']['name'].lower().startswith('conv'):
                model_list.append([layer['class_name'], layer['config']['name'],
                                   {'kernel_size':layer['config']['kernel_size'],
                                    'filters':layer['config']['filters']}])
            elif layer['class_name'] == 'GlobalMaxPooling2D':
                model_list.append([layer['class_name'],
                                   layer['config']['name'],
                                   {}])
            elif layer['class_name'] == 'Activation':
                model_list.append([layer['class_name'],
                                   layer['config']['name'],
                                   {}])

        return model_list

    def evolution_process(self):
        # TODO single model now population future
        new_config=gl_config.copy('ga_'+str(self.max_ind))
        self.max_ind+=1
        before_model=self.population[0]
        evolution_choice_list = ['deeper']#, 'wider','add_skip']
        evolution_choice = evolution_choice_list[random.randint(0, len(evolution_choice_list))]
        print evolution_choice

        if evolution_choice=='deeper':
            after_model=self.net2net.deeper(before_model,config=new_config)
        elif evolution_choice=='wider':
            after_model=self.net2net.wider(before_model,config=new_config)
        elif evolution_choice=='add_skip':
            after_model = self.net2net.add_skip(before_model, config=new_config)

        # TODO single model now population future
        self.population=[after_model]

    def fit_model_process(self):
        # TODO parallel
        for model in self.population:
            model.comp_fit_eval()


    def genetic_grow_model(self):
        Utils.mkdir_p('ga/')
        model_l=self.get_model_list(self.make_init_model())
        graph = MyGraph(model_l)

        self.population.append(MyModel(gl_config, graph))
        # TODO parallel
        gl_config.evoluation_time=10
        for i in range(gl_config.evoluation_time):
            self.fit_model_process()
            self.evolution_process()
            self.select_process()

    def select_process(self):
        pass


if __name__=="__main__":
    dbg = True
    if dbg:
        gl_config = MyConfig(epochs=0, verbose=2, dbg=dbg, name='ga',evoluation_time=2)
    else:
        gl_config = MyConfig(epochs=100, verbose=1, dbg=dbg, name='ga',evoluation_time=100)

    ga=GA()
    ga.genetic_grow_model()




