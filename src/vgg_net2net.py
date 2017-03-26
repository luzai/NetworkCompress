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


vgg_conv_width=[64,64,
                128,128,
                256,256,256,256]
vgg_fc_width=[2048,1024,10]


if __name__ == "__main__":

    args = parse_args()
    train_data, validation_data = load_data(args.dbg)

    if args.dbg:
        args.nb_epoch = 1
        args.gl_verbose = 2
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
    student_conv_width,student_fc_width=get_width(teacher_model)
    print student_conv_width,student_fc_width
    print vgg_conv_width,vgg_fc_width


    command = [
        "net2net",  # model name
        ["net2deeper", "conv2", 0, args.gl_verbose],

        ["net2wider", "conv3", 2, 0, args.gl_verbose],
        ["net2deeper", "conv3", 0, args.gl_verbose],

        ["net2deeper", "conv4", 0, args.gl_verbose],
        ["net2wider", "conv5", 2, 0, args.gl_verbose],
        ["net2deeper", "conv6", 0, args.gl_verbose],
        ["net2deeper", "conv7", 0, args.gl_verbose],
        ["net2deeper", "conv8", 0, args.gl_verbose],

        ["net2deeper", "fc1", args.nb_epoch, args.gl_verbose],
        ["net2wider", "fc1",int(2048./64), args.nb_epoch, args.gl_verbose],
        ["net2wider", "fc2", int(1024. / 64), args.nb_epoch, args.gl_verbose]

    ]
    student_model, log1 = make_model(teacher_model, command,
                                     train_data, validation_data)
    print get_width(student_model)