from net2net import *
from keras.layers import Dropout,LSTM,Embedding,Activation
from sklearn.preprocessing import MultiLabelBinarizer
from net2net  import *

mSequentialModel = Sequential()
mSequentialModel.add(Conv2D(64, 3, 3, input_shape=input_shape,
                 border_mode='same', name='conv1',activation='relu'))
mSequentialModel.add(MaxPooling2D(name='pool1'))
mSequentialModel.add(Dropout(0.25))
mSequentialModel.add(Conv2D(64, 3, 3, border_mode='same', name='conv2',activation='relu'))
mSequentialModel.add(MaxPooling2D(name='pool2'))
mSequentialModel.add(Dropout(0.25))

mSequentialModel.add(Flatten(name='flatten'))
mSequentialModel.add(Dense(64, activation='relu', name='fc1'))
mSequentialModel.add(Dropout(0.5))
mSequentialModel.add(Dense(nb_class, name='fc2'))

image_input=Input(shape=input_shape)
logits=mSequentialModel(image_input)
output=Activation('softmax')(logits)

model=Model(input=image_input,output=[logits,output])

model.compile(loss=['mean_squared_error','categorical_crossentropy'],
              loss_weights=[1,0],
              optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy'])
print([l.name for l in model.layers])

save_model_config(model,"sequential+functional")

model = Sequential()
model.add(Conv2D(64, 3, 3, input_shape=input_shape,
                 border_mode='same', name='conv1'))
model.add(MaxPooling2D(name='pool1'))
model.add(Conv2D(64, 3, 3, border_mode='same', name='conv2'))
model.add(MaxPooling2D(name='pool2'))
model.add(Flatten(name='flatten'))
model.add(Dense(64, activation='relu', name='fc1'))
model.add(Dense(nb_class, activation='softmax', name='fc2'))
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy'])
save_model_config(model,"sequential")
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