# -*- coding: utf-8 -*-
from keras.utils import np_utils
from keras.layers import add, Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, GlobalMaxPooling2D
from keras.layers.core import Activation
from keras.models import Model
from keras.optimizers import SGD
from keras.models import model_from_json
import pydot, graphviz
from keras.utils import plot_model
import os
import random
import json
import Config

'''
    define original init model 
'''
def make_init_model( ):
    input_data = Input(shape = (32, 32, 3))
    
    init_model_index = random.randint(1, 4)
    init_model_index = 2

    if init_model_index == 1:   # one conv layer with kernel num = 64
        stem_conv_1 = Conv2D(64, (1, 1), padding = 'same')(input_data)

    elif init_model_index == 2: # two conv layers with kernel num = 64
        stem_conv_1 = Conv2D(64, (1, 1), padding = 'same')(input_data)
        stem_conv_2 = Conv2D(64, (1, 1), padding = 'same')(stem_conv_1)

    elif init_model_index == 3: # one conv layer with a wider kernel num = 128 
        stem_conv_1 = Conv2D(128, (1, 1), padding = 'same')(input_data)

    elif init_model_index == 4: # two conv layers with a wider kernel_num = 128 
        stem_conv_1 = Conv2D(128, (1, 1), padding = 'same')(input_data)
        stem_conv_2 = Conv2D(128, (1, 1), padding = 'same')(stem_conv_1)

    stem_global_pooling_1 = GlobalMaxPooling2D()(stem_conv_1)
    stem_softmax_1 = Activation('softmax')(stem_global_pooling_1)

    model = Model(inputs = input_data, outputs=stem_softmax_1)
    return model

'''
    define residual block 
'''
def residual_block(input, pre_output_shape, idx):
    conv_layer = Conv2D(pre_output_shape, (1, 1), padding = 'same', name = 'res_conv_' + str(idx))(input)
    res_layer = add([input, conv_layer], name = 'res_add_' + str(idx))
    return res_layer


'''
    define inception block
'''
def inception_block( input_data, idx ):
    tower_1 = Conv2D(64, (1, 1), padding = 'same', activation = 'relu', name = 'incep_conv_1a_' + str(idx))(input_data)
    tower_1 = Conv2D(64 / 3, (3, 3), padding = 'same', activation = 'relu', name = 'incep_conv_1b_' + str(idx))(tower_1)

    tower_2 = Conv2D(64, (1, 1), padding = 'same', activation = 'relu', name = 'incep_conv_2a_' + str(idx))(input_data)
    tower_2 = Conv2D(64 / 3, (5, 5), padding = 'same', activation = 'relu', name = 'incep_conv_2b_' + str(idx))(tower_2)

    tower_3 = MaxPooling2D((3, 3), strides=(1,1), padding = 'same', name = 'incep_conv_3a_' + str(idx))(input_data)
    tower_3 = Conv2D(64 / 3 + 1, (1, 1), padding = 'same', activation = 'relu', name = 'incep_conv_3b_' + str(idx))(tower_3)

    output = concatenate([tower_1, tower_2, tower_3], axis = 3)

    return output 

'''
    get model blocks as a list from model
'''
def get_model_list(model):
    model_list = []
    model_dict = json.loads(model.to_json())

    with open('model_frame.json', 'w') as outfile:
        json.dump(model_dict, outfile)

    model_layer = model_dict['config']['layers']

    for layer in model_layer:
        layer_name = layer['config']['name']
        layer_output_shape = model.get_layer(layer_name).output_shape
        if layer['class_name'] == 'InputLayer':
            model_list.append([layer['class_name'], layer['config']['batch_input_shape'][1:]])
        elif layer['class_name'] == 'Conv2D' and layer['config']['name'].startswith('conv'):
            model_list.append([layer['class_name'], layer['config']['kernel_size'], layer['config']['filters']])
        elif layer['class_name'] == 'Add' and layer['config']['name'].startswith('res'):
            model_list.append(['ResidualBlock', layer_output_shape[3]])
        elif layer['class_name'] == 'Concatenate':
            model_list.append(['InceptionBlock', layer_output_shape[3]])
        elif layer['class_name'] == 'GlobalMaxPooling2D':
            model_list.append([layer['class_name']])
        elif layer['class_name'] == 'Activation':
            model_list.append([layer['class_name']])

    return model_list

'''
    Build the model according to model_list
'''
def Build(model_list):
    print model_list
    for idx, layer in enumerate(model_list):
        type = layer[0]
        if type == 'InputLayer':
            input = Input(shape = layer[1])
            x = input

        elif type == 'Conv2D':
            x = Conv2D(filters = layer[2], kernel_size = layer[1], padding = 'same')(x)

        elif type == 'InceptionBlock':
            x = inception_block(x, idx)
            
        elif type == 'ResidualBlock':
            x = residual_block(x, layer[1], idx)
            
        elif type=="GlobalMaxPooling2D":
            x = GlobalMaxPooling2D()(x)

        elif type=="Activation":
            x = Activation('softmax')(x)
        
    model = Model(inputs=input, outputs=x)
    return model


'''
    add a new conv layer into model_list
'''
#TODO: add maxpooling layer after conv-layer
def conv_deeper(model):
    model_list = get_model_list(model)
    
    for idx, layer in enumerate(model_list):
        if layer[0] == 'Conv2D' or layer[0] == 'InceptionBlock' or layer[0] == 'ResidualBlock':
            insert_idx = idx + 1

    model_list.insert(insert_idx, ['Conv2D', [1, 1], 64])
    
    new_model = Build(model_list)

    return new_model

'''
    change a conv layer's kernel width
'''
def conv_wider(model):
    model_list = get_model_list(model)
    
    for idx, layer in enumerate(model_list):
        if layer[0] == 'Conv2D':
             wider_layer = layer
             insert_idx = idx + 1
        
    # wider operation: filters * 2
    wider_layer[2] *= 2
        
    # if next layer is residual layer, we need to change residual layer's input shape
    while(model_list[insert_idx][0] == 'ResidualBlock'):
        model_list[insert_idx][1] = wider_layer[2]
        insert_idx = insert_idx + 1
    
    new_model = Build(model_list)

    return new_model

'''
    add a inception block into model_list
'''
def add_inception_block(model):
    model_list = get_model_list(model)
    
    for idx, layer in enumerate(model_list):
        if layer[0] == 'Conv2D' or layer[0] == 'InceptionBlock' or layer[0] == 'ResidualBlock':
            insert_idx = idx + 1

    model_list.insert(insert_idx, ['InceptionBlock'])
    
    new_model = Build(model_list)

    return new_model

'''
    add a residual block into model_list
'''
def add_skipping(model):
    model_list = get_model_list(model)
    
    insert_idx = -1
    #TODO: need to get the output shape from the last layer, use it as a parameter 
    for idx, layer in enumerate(model_list):
        if layer[0] == 'Conv2D' or layer[0] == 'InceptionBlock' or layer[0] == 'ResidualBlock':
            insert_idx = idx + 1
            if layer[0] == 'Conv2D':
                pre_output_shape = layer[2]
            else:
                pre_output_shape = layer[1]
    if insert_idx != -1:                                          
        model_list.insert(insert_idx, ['ResidualBlock', pre_output_shape])
    
    new_model = Build(model_list)

    return new_model


'''
    variation a model, return a variated new model
'''
#TODO: model-suite, by calculate fitness function to delete last k models and evolute top k models
def evolution_process(model):
    evolution_choice_list = ['conv_deeper', 'conv_wider', 'add_inception_block', 'add_skipping']
    evolution_choice = evolution_choice_list[random.randint(0, 3)]
    print evolution_choice

    if evolution_choice == 'conv_deeper':
        new_model = conv_deeper(model)
    elif evolution_choice == 'conv_wider':
        new_model = conv_wider(model)
    elif evolution_choice == 'add_inception_block':
        new_model = add_inception_block(model)
    elif evolution_choice == 'add_skipping':
        new_model = add_skipping(model)

    
    return new_model

'''
    use genetic algorithm to grow a convolution model
'''
import Utils
#TODO: maintain a model suite with number k
def genetic_grow_model():
    Utils.mkdir_p('model_grow/')
    model = make_init_model()
    model_name = 'inception_model_evolution_0'
    Utils.vis_model(model,'model_grow/' + model_name,show_shapes=False)

    evolution_time = 100
    
    for i in range(evolution_time):
        model = evolution_process(model)
        model_name = 'inception_model_evolution_' + str(i)
        print Utils.nvidia_smi()
        Utils.vis_model(model, name='model_grow/'+model_name, show_shapes=False)
        print Utils.count_weight(model)

if __name__ == "__main__":
    model = genetic_grow_model()
