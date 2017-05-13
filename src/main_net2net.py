from __future__ import print_function
from Net2Net import *


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



if __name__ == "__main__":

    args = parse_args()
    train_data, test_data = load_data(args.dbg)

    if args.dbg:
        args.nb_epoch = 1
        args.gl_verbose = 0
        args.nb_teacher_epoch = 1
        np.random.seed(16)

    pprint(args)

    '''train teacher model'''
    teacher_model, history = make_teacher_model(
        train_data, test_data,
        args.nb_teacher_epoch,
        args.gl_verbose
    )
    log0 = [[
        "make_teacher",
        [l.name for l in teacher_model.layers],
        # history.history["val_logits_acc"] if history.history else[],
        history.history["val_acc"] if history.history else[],
    ]]

    '''train net2net student model'''
    command = [
        "net2net",  # model name
        ["net2wider", "conv1", 2, args.nb_epoch, args.gl_verbose],  # command name, new layer, new width, number of epoch
        # ["net2wider", "conv2", 2, args.nb_epoch, args.gl_verbose],  # command name, new layer, new width, number of epoch
        ["net2wider", "fc1", 2, args.nb_epoch, args.gl_verbose],
        # ["net2deeper", "conv2", args.nb_epoch, args.gl_verbose],
        # ["net2deeper", "fc1", args.nb_epoch, args.gl_verbose]
    ]
    student_model, log1 = make_model(teacher_model, command,
                                     train_data, test_data)

    # '''train random student model'''
    # command = [
    #     ["random"],
    #     ["random-pad", "conv1", 2, 0, args.gl_verbose],
    #     ["random-pad", "fc1", 2, 0, args.gl_verbose],
    #     ["random-init", "conv2", 0, args.gl_verbose],
    #     ["random-init", "fc1", args.nb_epoch, args.gl_verbose]
    # ]
    # student_model, log2 = make_model(teacher_model, command,
    #                                  train_data, validation_data)

    '''print log'''
    map(lambda x: pprint(x, indent=2), ["\n", log0, "\n", log1])

    vis(log0, [log1],command)
