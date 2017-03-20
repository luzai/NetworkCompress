from net2net import *
from keras.layers import Dropout,LSTM,Embedding,Activation
from sklearn.preprocessing import MultiLabelBinarizer

DH = 200
N_WORDS = 100000
SEQ_LENGTH = 50
N_CLASSES = 10
N_EXAMPLES = 32

x = np.random.randint(N_WORDS, size=(N_EXAMPLES, SEQ_LENGTH))
x_test = np.random.randint(N_WORDS, size=(N_EXAMPLES, SEQ_LENGTH))
y = np.random.randint(N_CLASSES, size=(N_EXAMPLES, 1))
y = MultiLabelBinarizer().fit_transform(y)  # encode in one-hot

print('x.shape:', x.shape)
print('y.shape:', y.shape)

for i in range(10):
    print('ITERATION {}'.format(i))

    model = Sequential()
    model.add(Embedding(input_dim=N_WORDS, output_dim=DH, input_length=SEQ_LENGTH))
    model.add(Dropout(.2))
    model.add(LSTM(DH))
    model.add(Dropout(.5))
    model.add(Dense(N_CLASSES))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop',metrics=["accuracy"])

    model.fit(x, y, nb_epoch=1)
    predictions = model.predict(x_test)
    os.system("nvidia-smi")
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