from net2net import *


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='net work compression')
    parser.add_argument('--epoch', dest='nb_epoch', help='number epoch',
                        default=150, type=int)
    parser.add_argument("--teacher-epoch", dest="nb_teacher_epoch", help="number teacher epoch",
                        default=50, type=int)
    parser.add_argument('--verbose', dest='gl_verbose', help="global verbose",
                        default=1, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument("--dbg", dest="dbg",
                        help="for dbg",
                        action="store_true")
    parser.add_argument('--gpu', dest='gpu_id', help='gpu id',
                        default=0, type=int)
    _args = parser.parse_args()
    return _args


def rand_cmd(possible_layer):
    cmd = []
    if np.random.rand(1) > 0.5:
        cmd.append("net2wider")
        for i, v in enumerate(possible_layer):
            if re.findall(r"[a-z]+", v)[0] == "fc":
                break
        i-=1
        shield_layer=possible_layer[i]
        del possible_layer[i]
        shield_layer_fc=possible_layer[-1]
        del possible_layer[-1]
        cmd += [
            np.random.choice(possible_layer),
            np.random.uniform(1, 3),
            10000,
            args.gl_verbose
        ]
        possible_layer.insert(i,shield_layer)
        possible_layer.append(shield_layer_fc)
    else:
        cmd.append("net2deeper")
        new_layer = np.random.choice(possible_layer)
        cmd += [
            new_layer,
            10000,
            args.gl_verbose
        ]
        i = -1
        for i_, v in enumerate(possible_layer):
            if v == new_layer:
                i = i_
                break
        # layer_type = re.findall(r"[a-z]+", new_layer)[0]
        # layer_ind = int(re.findall(r"\d+", new_layer)[0])
        possible_layer.insert(i + 1, new_layer+"_1")
        possible_layer=reorder_list(possible_layer)
    return possible_layer, cmd


if __name__ == "__main__":

    args = parse_args()
    train_data, validation_data = load_data(args.dbg)
    # transfer_train_data,transfer_validation_data=get_transfer_data("../data/transfer_data/")

    if args.dbg:
        args.nb_epoch = 1
        args.gl_verbose = 2
        args.nb_teacher_epoch = 10000

    pprint(args)

    """train teacher model"""
    teacher_model, history = make_teacher_model(
        train_data, validation_data,
        args.nb_teacher_epoch,
        args.gl_verbose
    )

    log0 = [[
        ["make_teacher"],
        [l.name for l in teacher_model.layers],
        history.history["val_acc"] if history.history else[],
    ]]

    '''train net2net student model'''
    layer_names = [l.name for l in teacher_model.layers]
    possible_layer=[]
    for layer in layer_names:
        layer_type=re.findall(r"[a-z]+",layer)
        if len(layer_type)>0 \
                and layer_type[0]=="conv" or layer_type[0]=="fc":
                possible_layer+=[layer]

    command = ["rand_net2net"]
    for I in range(18):
        possible_layer, cmd = rand_cmd(possible_layer)
        command.append(cmd)

    pprint(command)
    print(possible_layer)

    # command = [
    #     ["net2net"],  # model name
    #     ["net2wider", "conv1", 2, 0, args.gl_verbose],  # command name, new layer, new width, number of epoch
    #     ["net2wider", "fc1", 2, 0, args.gl_verbose],
    #     ["net2deeper", "conv2", 0, args.gl_verbose],
    #     ["net2deeper", "fc1", args.nb_epoch, args.gl_verbose]
    # ]

    student_model, log1 = make_model(teacher_model, command,
                                     train_data, validation_data)
    print log1