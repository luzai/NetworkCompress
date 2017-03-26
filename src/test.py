from net2net import *
from keras.layers import Dropout,LSTM,Embedding,Activation,Input
from keras.models import Model
from sklearn.preprocessing import MultiLabelBinarizer
from net2net  import *

image_input = Input(shape=input_shape)

conv1 = Conv2D(64, 3, 3,
               border_mode='same', name='conv1', activation='relu')(image_input)
pool1 = MaxPooling2D(name='pool1')(conv1)
drop1 = Dropout(0.25)(pool1)

conv2 = Conv2D(64, 3, 3, border_mode='same', name='conv2', activation='relu')(drop1)
pool2 = MaxPooling2D(name='pool2')(conv2)
drop2 = Dropout(0.25)(pool2)

flatten = Flatten(name='flatten')(drop2)
fc1 = Dense(64, activation='relu', name='fc1')(flatten)
fc1_drop1 = Dropout(0.5)(fc1)
logits = Dense(nb_class, name='fc2')(fc1_drop1)
output = Activation('softmax')(logits)

model = Model(input=image_input, output=[logits, output])

model.compile(loss=['mean_squared_error', 'categorical_crossentropy'],
              loss_weights=[1, 0],
              optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy'])
model.layers[1].name="TTT"
model.layers[1].input=model.layers[0].name
# model.layers[1].output
print([l.name for l in model.layers])
print(model.get_layer("fc2"))
save_model_config(model,"functional")

def smooth(x,y):
    print len(x),len(y)
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter
    import numpy as np
    x,y=np.array(x),np.array(y)
    # xx = np.linspace(x.min(), x.max(), x.shape[0])
    xx=x
    # interpolate + smooth
    itp = interp1d(x, y, kind='linear')
    window_size, poly_order = 101, 3
    yy_sg = savgol_filter(itp(xx), window_size, poly_order)
    return xx,yy_sg