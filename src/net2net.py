# !/usr/bin/env python
from __init__ import *

from keras.models import Sequential, model_from_json
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.optimizers import SGD
from keras.utils import np_utils
# from keras.datasets import mnist
from keras.datasets import cifar10
from keras.utils import visualize_util

input_shape = (3, 32, 32)  # image shape
nb_class = 10  # number of class


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--epoch', dest='nb_epoch', help='number epoch',
                        default=4, type=int)
    parser.add_argument("--teacher-epoch",dest="nb_teacher_epoch",help="number teacher epoch",
                        default=50,type=int)
    parser.add_argument('--verbose', dest='gl_verbose', help="global verbose",
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument("--dbg", dest="dbg",
                        help="for dbg",
                        action="store_true")
    parser.add_argument("--per_iter",dest="per_iter",
                        help="per_iter",
                        action="store_true")
    parser.add_argument('--gpu', dest='gpu_id', help='gpu id',
                        default=0, type=int)

    # parser.add_argument('--verbose', dest='demo_net', help='Network to use [vgg16]',
    #                     choices=NETS.keys(), default='vgg16')

    args = parser.parse_args()

    return args


# load and pre-process data
def preprocess_input(x):
    return x.reshape((-1,) + input_shape) / 255.


def preprocess_output(y):
    return np_utils.to_categorical(y,nb_class)


# --------
# prepare data
# --------

def limit_data(train_x):
    return train_x[:200]


class AccHist(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.accs = []

    def on_batch_end(self, batch, logs={}):
        # pass
        if args.per_iter:
            self.accs.append(logs.get('acc'))
        else:
            self.accs=[]



# knowledge transfer algorithms
def wider_conv2d_weight(teacher_w1, teacher_b1, teacher_w2, new_width, init):
    '''Get initial weights for a wider conv2d layer with a bigger nb_filter,
    by 'random-pad' or 'net2wider'.

    # Arguments
        teacher_w1: `weight` of conv2d layer to become wider,
          of shape (nb_filter1, nb_channel1, kh1, kw1)
        teacher_b1: `bias` of conv2d layer to become wider,
          of shape (nb_filter1, )
        teacher_w2: `weight` of next connected conv2d layer,
          of shape (nb_filter2, nb_channel2, kh2, kw2)
        new_width: new `nb_filter` for the wider conv2d layer
        init: initialization algorithm for new weights,
          either 'random-pad' or 'net2wider'
    '''
    assert teacher_w1.shape[0] == teacher_w2.shape[1], (
        'successive layers from teacher model should have compatible shapes')
    assert teacher_w1.shape[0] == teacher_b1.shape[0], (
        'weight and bias from same layer should have compatible shapes')
    assert new_width > teacher_w1.shape[0], (
        'new width (nb_filter) should be bigger than the existing one')

    n = new_width - teacher_w1.shape[0]
    if init == 'random-pad':
        new_w1 = np.random.normal(0, 0.1, size=(n,) + teacher_w1.shape[1:])
        new_b1 = np.ones(n) * 0.1
        new_w2 = np.random.normal(0, 0.1, size=(
                                                   teacher_w2.shape[0], n) + teacher_w2.shape[2:])
    elif init == 'net2wider':
        index = np.random.randint(teacher_w1.shape[0], size=n)
        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[index, :, :, :]
        new_b1 = teacher_b1[index]
        new_w2 = teacher_w2[:, index, :, :] / factors.reshape((1, -1, 1, 1))
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)

    student_w1 = np.concatenate((teacher_w1, new_w1), axis=0)
    if init == 'random-pad':
        student_w2 = np.concatenate((teacher_w2, new_w2), axis=1)
    elif init == 'net2wider':
        # add small noise to break symmetry, so that student model will have
        # full capacity later
        noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
        student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=1)
        student_w2[:, index, :, :] = new_w2
    student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

    return student_w1, student_b1, student_w2


def wider_fc_weight(teacher_w1, teacher_b1, teacher_w2, new_width, init):
    '''Get initial weights for a wider fully connected (dense) layer
       with a bigger nout, by 'random-padding' or 'net2wider'.

    # Arguments
        teacher_w1: `weight` of fc layer to become wider,
          of shape (nin1, nout1)
        teacher_b1: `bias` of fc layer to become wider,
          of shape (nout1, )
        teacher_w2: `weight` of next connected fc layer,
          of shape (nin2, nout2)
        new_width: new `nout` for the wider fc layer
        init: initialization algorithm for new weights,
          either 'random-pad' or 'net2wider'
    '''
    assert teacher_w1.shape[1] == teacher_w2.shape[0], (
        'successive layers from teacher model should have compatible shapes')
    assert teacher_w1.shape[1] == teacher_b1.shape[0], (
        'weight and bias from same layer should have compatible shapes')
    assert new_width > teacher_w1.shape[1], (
        'new width (nout) should be bigger than the existing one')

    n = new_width - teacher_w1.shape[1]
    if init == 'random-pad':
        new_w1 = np.random.normal(0, 0.1, size=(teacher_w1.shape[0], n))
        new_b1 = np.ones(n) * 0.1
        new_w2 = np.random.normal(0, 0.1, size=(n, teacher_w2.shape[1]))
    elif init == 'net2wider':
        index = np.random.randint(teacher_w1.shape[1], size=n)
        factors = np.bincount(index)[index] + 1.
        new_w1 = teacher_w1[:, index]
        new_b1 = teacher_b1[index]
        new_w2 = teacher_w2[index, :] / factors[:, np.newaxis]
    else:
        raise ValueError('Unsupported weight initializer: %s' % init)

    student_w1 = np.concatenate((teacher_w1, new_w1), axis=1)
    if init == 'random-pad':
        student_w2 = np.concatenate((teacher_w2, new_w2), axis=0)
    elif init == 'net2wider':
        # add small noise to break symmetry, so that student model will have
        # full capacity later
        noise = np.random.normal(0, 5e-2 * new_w2.std(), size=new_w2.shape)
        student_w2 = np.concatenate((teacher_w2, new_w2 + noise), axis=0)
        student_w2[index, :] = new_w2
    student_b1 = np.concatenate((teacher_b1, new_b1), axis=0)

    return student_w1, student_b1, student_w2


def deeper_conv2d_weight(teacher_w):
    '''Get initial weights for a deeper conv2d layer by net2deeper'.

    # Arguments
        teacher_w: `weight` of previous conv2d layer,
          of shape (nb_filter, nb_channel, kh, kw)
    '''
    nb_filter, nb_channel, kh, kw = teacher_w.shape
    student_w = np.zeros((nb_filter, nb_filter, kh, kw))
    for i in xrange(nb_filter):
        student_w[i, i, (kh - 1) / 2, (kw - 1) / 2] = 1.
    student_b = np.zeros(nb_filter)
    return student_w, student_b


def copy_weights(teacher_model, student_model, layer_names=None):
    if layer_names is None:
        layer_names = [l.name for l in teacher_model.layers]
    for name in layer_names:
        weights = teacher_model.get_layer(name=name).get_weights()
        try:
            student_model.get_layer(name=name).set_weights(weights)
        except ValueError as e:
            # print "some layer shape change, Don't copy, We Init it Manually! OK!"
            # print "\t It is OK! Detail: {}".format(e)
            pass


            # methods to construct teacher_model and student_models
    '''Copy weights from teacher_model to student_model,
     for layers with names listed in layer_names
    '''


def make_teacher_model(train_data, validation_data, nb_epoch=3):
    '''Train a simple CNN as teacher model.
    '''
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
    print([l.name for l in model.layers])
    acc_hist = AccHist()
    history = model.fit(
        *(train_data),
        nb_epoch=nb_epoch,
        validation_data=validation_data,
        verbose=gl_verbose if nb_epoch != 0 else 0,
        callbacks=[acc_hist]
    )
    # print acc_hist.accs
    return model, history, [t[()] for t in acc_hist.accs]


def wider_conv2d_dict(new_conv1_width, modify_layer_name, teacher_model_dict):
    student_model_dict = copy.deepcopy(teacher_model_dict)
    # student_model_dict=teacher_model_dict
    name2ind = {student_model_dict["config"][i]["config"]["name"]: i
                for i in range(len(student_model_dict["config"]))}
    ind2name = [student_model_dict["config"][i]["config"]["name"]
                for i in range(len(student_model_dict["config"]))]

    modify_layer_ind = name2ind[modify_layer_name]
    student_model_dict["config"][modify_layer_ind]["config"]["nb_filter"] = new_conv1_width
    # TODO next layer is not conv
    # student_model_dict["config"][modify_layer_ind+1]["config"]["batch_input_shape"][1]=new_conv1_width
    return student_model_dict


def wider_fc_dict(new_fc1_width, modify_layer_name, teacher_model_dict):
    # student_model_dict=teacher_model_dict
    student_model_dict = copy.deepcopy(teacher_model_dict)
    name2ind = {student_model_dict["config"][i]["config"]["name"]: i
                for i in
                range(len(student_model_dict["config"]))}
    ind2name = [student_model_dict["config"][i]["config"]["name"]
                for i in range(len(student_model_dict["config"]))]
    modify_layer_ind = name2ind[modify_layer_name]
    student_model_dict["config"][modify_layer_ind]["config"]["output_dim"] = new_fc1_width
    # automated!
    # student_model_dict["config"][modify_layer_ind]["config"]["input_dim"]=new_fc1_width

    return student_model_dict


def reorder_dict(d):
    len_of_d = len(d["config"])
    name2ind = {d["config"][i]["config"]["name"]: i
                for i in range(len(d["config"]))}
    names = [d["config"][i]["config"]["name"]
             for i in range(len(d["config"]))]

    for i, v in enumerate(names):
        if re.findall(r"_", v): break
    type = re.findall(r"[a-z]+", names[i])[0]
    # ind=re.findall(r"\d+",names[i])[0]
    # ind=int(ind)
    start = 1
    for i, v in enumerate(names):
        now_type = re.findall(r"[a-z]+", v)[0]
        if now_type == type:
            names[i] = type + str(start)
            start += 1
    for i in range(len_of_d):
        d["config"][i]["config"]["name"] = names[i]
    return d


def reorder_model(model):
    names = [l.name for l in model.layers]
    len_of_model = len(names)
    names2ind = {v: i for i, v in enumerate(names)}

    for i, v in enumerate(names):
        if re.findall(r"_", v): break
    type = re.findall(r"[a-z]+", names[i])[0]
    start = 1
    for i, v in enumerate(names):
        now_type = re.findall(r"[a-z]+", v)[0]
        if now_type == type:
            names[i] = type + str(start)
            start += 1
    for i, v in enumerate(names):
        model.layers[i].name = v
    return model


def make_wider_student_model(teacher_model, train_data,
                             validation_data, init, new_name, new_width, nb_epoch=3):
    '''Train a wider student model based on teacher_model,
       with either 'random-pad' (baseline) or 'net2wider'
    '''
    new_type = re.findall(r"[a-z]+", new_name)[0]
    new_ind = re.findall(r"\d+", new_name)[0]
    new_ind = int(new_ind)
    teacher_model_dict = json.loads(teacher_model.to_json())
    if new_type == "conv":
        new_conv1_width = new_width
        new_conv1_name = new_name
        student_model_dict = wider_conv2d_dict(new_conv1_width, new_conv1_name, teacher_model_dict)

    elif new_type == "fc":
        new_fc1_width = new_width
        new_fc1_name = new_name
        student_model_dict = wider_fc_dict(new_fc1_width, new_fc1_name, teacher_model_dict)

    model = model_from_json(json.dumps(student_model_dict))

    layer_name = [l.name for l in teacher_model.layers]
    # layer_name=set(layer_name)
    # layer_name.discard(new_conv1_name)
    # layer_name.discard(new_fc1_name)
    copy_weights(teacher_model, model, layer_name)
    if new_type == "conv":
        next_new_name = new_type + str(new_ind + 1)
        w_conv1, b_conv1 = teacher_model.get_layer(new_name).get_weights()
        w_conv2, b_conv2 = teacher_model.get_layer(next_new_name).get_weights()
        new_w_conv1, new_b_conv1, new_w_conv2 = wider_conv2d_weight(
            w_conv1, b_conv1, w_conv2, new_conv1_width, init)
        model.get_layer(new_name).set_weights([new_w_conv1, new_b_conv1])
        model.get_layer(next_new_name).set_weights([new_w_conv2, b_conv2])
    elif new_type == "fc":
        next_new_name = new_type + str(new_ind + 1)
        w_fc1, b_fc1 = teacher_model.get_layer(new_name).get_weights()
        w_fc2, b_fc2 = teacher_model.get_layer(next_new_name).get_weights()
        new_w_fc1, new_b_fc1, new_w_fc2 = wider_fc_weight(
            w_fc1, b_fc1, w_fc2, new_fc1_width, init)
        model.get_layer(new_name).set_weights([new_w_fc1, new_b_fc1])
        model.get_layer(next_new_name).set_weights([new_w_fc2, b_fc2])

    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, momentum=0.9),
                  metrics=['accuracy'])
    print([l.name for l in model.layers])
    acc_hist = AccHist()
    history = model.fit(
        *(train_data),
        nb_epoch=nb_epoch,
        validation_data=validation_data,
        verbose=gl_verbose if nb_epoch != 0 else 0,
        callbacks=[acc_hist]
    )

    return model, history, [t[()] for t in acc_hist.accs]


def make_deeper_student_model(teacher_model, train_data,
                              validation_data, init, new_name, nb_epoch=3):
    '''Train a deeper student model based on teacher_model,
       with either 'random-init' (baseline) or 'net2deeper'
    '''
    new_type = re.findall(r"[a-z]+", new_name)[0]
    new_ind = re.findall(r"\d+", new_name)[0]
    new_ind = int(new_ind)
    teacher_model_dict = json.loads(teacher_model.to_json())
    student_model_dict = copy.deepcopy(teacher_model_dict)

    name2ind = {student_model_dict["config"][i]["config"]["name"]: i
                for i in range(len(student_model_dict["config"]))}
    ind2name = [student_model_dict["config"][i]["config"]["name"]
                for i in range(len(student_model_dict["config"]))]
    student_new_layer = copy.deepcopy(student_model_dict["config"][name2ind[new_name]])
    student_new_layer["config"]["name"] = new_name + "_1"
    if new_type == "conv":
        # student_new_layer["config"]["nb_filter"] NO NEED
        pass
    elif new_type == "fc":
        student_new_layer["config"]["init"] = "identity"
    student_model_dict["config"].insert(name2ind[new_name] + 1, student_new_layer)
    # student_model_dict=reorder_dict(student_model_dict)

    model = model_from_json(json.dumps(student_model_dict))

    if new_type == "conv" and init == 'net2deeper':
        prev_w, _ = teacher_model.get_layer(new_name).get_weights()
        new_weights = deeper_conv2d_weight(prev_w)
        model.get_layer(new_name + "_1").set_weights(new_weights)

    ## copy weights for other layers
    copy_weights(teacher_model, model)
    # print [l.name for l in model.layers]
    model = reorder_model(model)
    # print [l.name for l in model.layers]
    model.compile(loss='categorical_crossentropy',
                  optimizer=SGD(lr=0.001, momentum=0.9),
                  metrics=['accuracy'])
    print([l.name for l in model.layers])
    acc_hist = AccHist()
    history = model.fit(
        *(train_data),
        nb_epoch=nb_epoch,
        validation_data=validation_data,
        verbose=gl_verbose if nb_epoch != 0 else 0,
        callbacks=[acc_hist]
    )

    return model, history, [t[()] for t in acc_hist.accs]


def echo():
    print "verbose", gl_verbose, "epoch", nb_epoch,"teacher epoch",args.nb_teacher_epoch


def copy_model(model):
    model_dict = json.loads(model.to_json())
    new_model = model_from_json(json.dumps(model_dict))
    copy_weights(model, new_model)
    new_model.compile(loss='categorical_crossentropy',
                      optimizer=SGD(lr=0.001, momentum=0.9),
                      metrics=['accuracy'])
    return new_model


def make_model(teacher_model, commands):
    student_model = copy_model(teacher_model)
    log = []
    print "\n\n------------------------------\n\n"

    for cmd in commands[1:]:
        if cmd[0] == "net2wider" or cmd[0] == "random-pad":
            student_model, history, acc = make_wider_student_model(
                student_model,
                train_data, validation_data,
                *cmd
            )
        elif cmd[0] == "net2deeper" or cmd[0] == "random-init":
            student_model, history, acc = make_deeper_student_model(
                student_model,
                train_data, validation_data,
                *cmd
            )
        else:
            raise ValueError('Unsupported cmd: %s' % cmd[0])

        log_append_t=[
            cmd,
            [l.name for l in student_model.layers],
            history.history["val_acc"] if history.history else []
            ]
        if args.per_iter:
            log_append_t+=[acc]
        log.append(log_append_t)

    name = commands[0][0]
    os.chdir("./output")
    student_model.save_weights(name + ".h5")
    with open(name + ".json", "w") as f:
        json.dump(
            json.loads(student_model.to_json()),
            f,
            indent=2
        )
    with open(name + ".pkl", 'w') as f:
        cPickle.dump(log, f)
    with open(name + ".log", "w") as f:
        for log_item in log:
            f.write(str(log_item[0]) + "\n")
            [f.write("\t" + str(log_item_item) + "\n") for log_item_item in log_item[1:]]
    visualize_util.plot(teacher_model, to_file="teacher.png", show_shapes=True)
    visualize_util.plot(student_model, to_file=name + '.png', show_shapes=True)
    os.chdir("..")

    return student_model, log


def smooth(x, y):
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    from scipy.interpolate import interp1d
    from scipy.signal import savgol_filter
    import numpy as np
    x, y = np.array(x), np.array(y)
    xx = np.linspace(x.min(), x.max(), x.shape[0] + 100)
    # xx=x
    # interpolate + smooth
    itp = interp1d(x, y, kind='linear')
    window_size, poly_order = 101, 3
    yy_sg = savgol_filter(itp(xx), window_size, poly_order)
    return xx, yy_sg


def vis(log0, log12):

    acc0 = np.array(log0[0][-1])

    for log in log12:
        acc = []
        for log_item in log:
            acc += log_item[-1]
        acc = np.array(acc)
        acc_con = np.concatenate((acc0, acc))
        if args.per_iter:
            plt.plot(*smooth(np.arange(acc_con.shape[0]), acc_con))
        else:
            plt.plot(acc_con)
    plt.legend(["net2net", "random","origin"])
    # plt.plot(acc0)
    plt.savefig('benchmark.png')
    plt.show()



if __name__ == "__main__":
    global nb_epoch, gl_verbose, train_x, train_y, validation_x, validation_y
    (train_x, train_y), (validation_x, validation_y) = cifar10.load_data()
    print('train_x shape:', train_x.shape, 'train_y shape:', train_y.shape)
    print('validation_x shape:', validation_x.shape,
          'validation_y shape', validation_y.shape)
    train_x, validation_x = map(preprocess_input, [train_x, validation_x])
    train_y, validation_y = map(preprocess_output, [train_y, validation_y])

    print('Loading data...')
    print('train_x shape:', train_x.shape, 'train_y shape:', train_y.shape)
    print('validation_x shape:', validation_x.shape,
          'validation_y shape', validation_y.shape)
    print("\n\n---------------------------------------\n\n")

    args = parse_args()
    nb_epoch = args.nb_epoch
    gl_verbose = args.gl_verbose
    if args.dbg:
        train_x, train_y, validation_x, validation_y = map(limit_data, [train_x, train_y, validation_x, validation_y])
        nb_epoch = 150
        gl_verbose = 1
        args.nb_teacher_epoch=50
    echo()

    train_data = (train_x, train_y)
    validation_data = (validation_x, validation_y)
    ## train teacher model
    teacher_model, history, acc = make_teacher_model(train_data,
                                                     validation_data,
                                                     nb_epoch=args.nb_teacher_epoch)
    log0 = [[["make_teacher"],
            [l.name for l in teacher_model.layers],
            history.history["val_acc"] if history.history else[],
            ]]
    if args.per_iter:
        log0+=[acc]
    ## train net2net student model
    command = [
        ["net2net"],  # model name
        ["net2wider", "conv1", 128, 0],  # command name, new layer, new width, number of epoch
        ["net2wider", "fc1", 128, 0],
        ["net2deeper", "conv2", 0],
        ["net2deeper", "fc1", nb_epoch]
    ]
    student_model, log1 = make_model(teacher_model, command)


    ## train random student model
    command = [
        ["random"],
        ["random-pad", "conv1", 128, 0],
        ["random-pad", "fc1", 128, 0],
        ["random-init", "conv2", 0],
        ["random-init", "fc1", nb_epoch]
    ]
    student_model, log2 = make_model(teacher_model, command)


    ## continue train teacher model
    acc_hist = AccHist()
    history = teacher_model.fit(
        *(train_data),
        nb_epoch=nb_epoch,
        validation_data=validation_data,
        verbose=gl_verbose if nb_epoch != 0 else 0,
        callbacks=[acc_hist]
    )
    log3= [[["make_teacher_cont"],
            [l.name for l in teacher_model.layers],
            history.history["val_acc"] if history.history else[],
            ]]
    if args.per_iter:
        log3 += [t[()] for t in acc_hist.accs]

    ## print log
    # map(lambda x: pprint(x,indent=2),["\n",log0,"\n",log1,"\n",log2])

    vis(log0, [log1, log2,log3])
