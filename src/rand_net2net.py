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
    if np.random.rand(1) > 0.6:
        cmd.append("net2wider")
        for i, v in enumerate(possible_layer):
            if re.findall(r"[a-z]+", v)[0] == "fc":
                break
        i -= 1
        shield_layer = possible_layer[i]
        del possible_layer[i]
        shield_layer_fc = possible_layer[-1]
        del possible_layer[-1]
        cmd += [
            np.random.choice(possible_layer),
            np.random.uniform(1.2, 2.4),
            10000,
            args.gl_verbose
        ]
        possible_layer.insert(i, shield_layer)
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
        possible_layer.insert(i + 1, "^_"+new_layer )
        possible_layer = reorder_list(possible_layer)
    return possible_layer, cmd


if __name__ == "__main__":

    args = parse_args()
    train_data, validation_data = load_data(args.dbg)

    if args.dbg:
        args.nb_epoch = 1
        args.gl_verbose = 0
        args.nb_teacher_epoch = 1
        np.random.seed(16)
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
    for II in range(10):
        '''train net2net student model'''
        layer_names = [l.name for l in teacher_model.layers]
        possible_layer = []
        for layer in layer_names:
            layer_type = re.findall(r"[a-z]+", layer)
            if len(layer_type) > 0 \
                    and layer_type[0] == "conv" or layer_type[0] == "fc":
                possible_layer += [layer]

        command = [re.sub(
            r"\s",
            "_",
            datetime.datetime.now().ctime()
        )]

        for I in range(24):
            possible_layer, cmd = rand_cmd(possible_layer)
            command.append(cmd)

        pprint(command)
        print(possible_layer)

        student_model, log1 = make_model(teacher_model, command,
                                         train_data, validation_data)
        pprint(log1)
        vis(log0, [log1], command)
