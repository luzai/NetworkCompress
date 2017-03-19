from __init__ import *
from net2net import *
from load_transfer_data import get_transfer_data

# get_transfer_data("../data/transfer_data/")

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
    train_data, validation_data = load_data(args.dbg)
    if args.dbg:
        args.nb_epoch = 150
        args.gl_verbose = 1
        args.nb_teacher_epoch = 50

    pprint(args)

    ## train teacher model
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

    ## train net2net student model
    command = [
        ["net2net"],  # model name
        ["net2wider", "conv1", 2, 0, args.gl_verbose],  # command name, new layer, new width, number of epoch
        ["net2wider", "fc1", 2, 0, args.gl_verbose],
        ["net2deeper", "conv2", 0, args.gl_verbose],
        ["net2deeper", "fc1", args.nb_epoch, args.gl_verbose]
    ]
    student_model, log1 = make_model(teacher_model, command,
                                     train_data, validation_data)

    ## train random student model
    command = [
        ["random"],
        ["random-pad", "conv1", 2, 0, args.gl_verbose],
        ["random-pad", "fc1", 2, 0, args.gl_verbose],
        ["random-init", "conv2", 0, args.gl_verbose],
        ["random-init", "fc1", args.nb_epoch, args.gl_verbose]
    ]
    student_model, log2 = make_model(teacher_model, command,
                                     train_data, validation_data)

    ## continue train teacher model
    history = teacher_model.fit(
        *(train_data),
        nb_epoch=args.nb_epoch,
        validation_data=validation_data,
        verbose=args.gl_verbose if args.nb_epoch != 0 else 0,
        callbacks=[lr_reducer, early_stopper, csv_logger]
    )
    log3 = [[
        ["make_teacher_cont"],
        [l.name for l in teacher_model.layers],
        history.history["val_acc"] if history.history else[],
    ]]

    ## print log
    # map(lambda x: pprint(x,indent=2),["\n",log0,"\n",log1,"\n",log2])

    vis(log0, [log1, log2, log3])
