import time

import keras.layers
from keras.backend import tensorflow_backend as ktf
from keras.callbacks import TensorBoard
from keras.engine.topology import Layer
from keras.layers import Conv2D, Input, MaxPooling2D, Activation, GlobalMaxPooling2D

from Config import MyConfig
from Logger import logger


class CP2D(Layer):
    def __init__(self, output_dim, kernel_size=3, **kwargs):
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        super(CP2D, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(CP2D, self).build(input_shape)

    def call(self, x):
        return MaxPooling2D(name=self.name + '_pool')(
            Conv2D(self.output_dim, self.kernel_size, name=self.name + '_conv')(x))

    def compute_output_shape(self, input_shape):
        _input = Input(batch_shape=input_shape)
        _tensor = MaxPooling2D()(
            Conv2D(self.output_dim, self.kernel_size)(_input))
        return ktf.int_shape(_tensor)


setattr(keras.layers, 'CP2D', CP2D)

config = MyConfig(epochs=1, verbose=2, limit_data=True, name='ga', evoluation_time=10)

input = Input(shape=config.input_shape)
stem_conv_0 = Conv2D(120, 3, padding='same', name='conv2d1')(input)
stem_conv_1 = Conv2D(60, 3, padding='same', name='conv2d2')(stem_conv_0)
stem_cp = CP2D(10, 3, name='cp2d1')(stem_conv_1)
stem_global_pooling_1 = GlobalMaxPooling2D(name='globalmaxpooling2d1')(stem_cp)
stem_softmax_1 = Activation('softmax', name='activation1')(stem_global_pooling_1)
model = keras.models.Model(inputs=input, outputs=stem_softmax_1)

model.summary()
model.compile(optimizer='adam',  # rmsprop
              loss='categorical_crossentropy',
              metrics=['accuracy'])

logger.info("Start train model {}\n".format(config.name))
tic = time.time()
hist = model.fit(config.dataset['train_x'],
                 config.dataset['train_y'],
                 validation_data=(config.dataset['test_x'], config.dataset['test_y']),
                 verbose=config.verbose,
                 batch_size=config.batch_size,
                 epochs=config.epochs,
                 callbacks=[config.lr_reducer, config.csv_logger, config.early_stopper,
                            TensorBoard(log_dir=config.tf_log_path)]
                 )
logger.info("Fit model {} Consume {}:".format(config.name, time.time() - tic))
