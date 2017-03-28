from net2net import *

train_data, test_data = load_data(True)

image_input = Input(shape=input_shape)

conv1 = Conv2D(64, 3, 3,
               border_mode='same', name='conv1', activation='relu')(image_input)
pool1 = MaxPooling2D(name='pool1')(conv1)
drop1 = Dropout(0.25,name='drop1')(pool1)

conv2 = Conv2D(64, 3, 3, border_mode='same', name='conv2', activation='relu')(drop1)
pool2 = MaxPooling2D(name='pool2')(conv2)
drop2 = Dropout(0.25)(pool2)

flatten = Flatten(name='flatten')(drop2)
fc1 = Dense(64, activation='relu', name='fc1')(flatten)
fc1_drop1 = Dropout(0.5)(fc1)
logits = Dense(nb_class, name='logits')(fc1_drop1)
output = Activation('softmax', name='softmax')(logits)

model = Model(input=image_input, output=output)

model.compile(loss=['categorical_crossentropy'],
              loss_weights=[1],
              optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy'])
print([l.name for l in model.layers])
history = model.fit(
    train_data[0], train_data[1],
    nb_epoch=200,
    validation_data=test_data,
    verbose=2,
    callbacks=[lr_reducer, early_stopper, csv_logger],
    batch_size=batch_size
)
print history.history

func=K.function(inputs=[model.input,K.learning_phase()], outputs=[model.get_layer("conv2").output])
x = func([np.random.rand(*(1,3,32,32)),1]) # ok
x = func([image_input,1])[0] # error TypeError: ('Bad input argument to theano function with name "/usr/lib/python2.7/site-packages/keras/backend/theano_backend.py:955"  at index 0(0-based)', 'Expected an array-like object, but found a Variable: maybe you are trying to call a function on a (possibly shared) variable instead of a numeric array?')

x = Conv2D(64, 3, 3, border_mode='same', name='conv2', activation='relu')(x)
x = Dropout(0.25)(x)
output = K.function(inputs=[model.get_layer("flatten").input,K.learning_phase()], outputs=[model.output])(x)

new_model = Model(input=image_input, output=output)
new_model.compile(loss=['categorical_crossentropy'],
              loss_weights=[1],
              optimizer=SGD(lr=0.01, momentum=0.9),
              metrics=['accuracy'])
print([l.name for l in new_model.layers])
history = new_model.fit(
    train_data[0], train_data[1],
    nb_epoch=200,
    validation_data=test_data,
    verbose=2,
    callbacks=[lr_reducer, early_stopper, csv_logger],
    batch_size=batch_size
)
save_model_config(new_model, "functional")


