from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot

from net2net import *
input_shape=(3,16,16)

def make_model_for_conv_fc(width,with_activation=False,with_max_pool=False,with_dropout=False):
    input_node=Input(shape=(input_shape),name="input")
    x=Convolution2D(width,3,3,name='conv8',border_mode="same"
                    ,activation='relu'if  with_activation else None
                    )(input_node)
    x=MaxPooling2D(name="pool1")(x)  if with_max_pool else x
    x=Dropout(0.25,name="drop1")(x) if with_dropout else x
    x=Flatten(name="flatten")(x)
    output_node=Dense(2048,name='fc1'
            ,activation='relu'if  with_activation else None
            )(x)
    model=Model(input=input_node,output=output_node)
    model.trainable=False
    return model
np.random.seed(1)
input_inst=np.random.random((1,)+input_shape)
new_width=128
model=make_model_for_conv_fc(64)
model2=make_model_for_conv_fc(new_width)

copy_weights(teacher_model=model,student_model=model2)
copy_weights(student_model=model2,teacher_model=model)

w_conv8, b_conv8 = model.get_layer("conv8").get_weights()
w_fc1, b_fc1 = model.get_layer("fc1").get_weights()

# new_w_conv1, new_b_conv1, new_w_conv2 = wider_conv2d_weight(
#     w_conv1, b_conv1, w_conv2, 256, "net2wider")

n= new_width - w_conv8.shape[0]
index=np.random.randint(w_conv8.shape[0], size=n)
factors=np.bincount(index)[index]+1
new_w1=w_conv8[index, ...]
noise=np.random.normal(0,5e-2*new_w1.std(),size=new_w1.shape)
new_w_conv1=np.concatenate([w_conv8, new_w1 + noise], axis=0)
new_b1=b_conv8[index]
noise=np.random.normal(0,5e-2*new_b1.std(),size=new_b1.shape)
new_b_conv1=np.concatenate([b_conv8, new_b1 + noise], axis=0)

index_fc = np.empty(shape=[0, ], dtype=int)
factor_fc = np.empty(shape=[0, ], dtype=int)
for i, j in zip(index.reshape((-1,)), factors.reshape((-1,))):
    start = i * 256
    end = (i + 1) * 256
    #     print index_fc.shape,np.arange(start,end).shape
    index_fc = np.concatenate([index_fc, np.arange(start, end)])

    factor_fc = np.concatenate([factor_fc,
                                j * np.ones(shape=(256,))])

new_w2 = w_fc1[index_fc, :] / factor_fc.reshape((-1, 1))
noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
new_w_fc1 = np.concatenate([w_fc1, new_w2 + noise], axis=0)
new_w_fc1[index_fc, :] = new_w2

model2.get_layer("conv8").set_weights([new_w_conv1, new_b_conv1])
model2.get_layer("fc1").set_weights([new_w_fc1, b_fc1])
output_inst = K.function(inputs=[model.input, K.learning_phase()], outputs=[model.output])([input_inst, 0])[0]

output_inst2 = K.function(inputs=[model2.input, K.learning_phase()], outputs=[model2.output])([input_inst, 0])[0]
print output_inst, '\n', output_inst2