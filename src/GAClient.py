import multiprocessing as mp
import argparse, os, sys
import subprocess
import os.path as osp

root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)


def run(queue, name, epochs=100, verbose=1):
    import keras
    import tensorflow as tf

    from Config import MyConfig
    from Model import MyModel

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        config = MyConfig(name=name, epochs=epochs, verbose=verbose, clean=False)
        # device = 0
        # with tf.device(device):
        model = MyModel(config=config, model=keras.models.load_model(config.model_path))
        print "start train {}".format(name)
        # with config.sess.as_default():
        score = model.comp_fit_eval()
        queue.put((name, score))


def run_self(queue, name, epochs=100, verbose=1):
    my_env = os.environ.copy()
    # todo
    child1 = subprocess.Popen(("python " + osp.join(root_dir, 'src', 'GAClient.py')
                               + " -n " + name +
                               " -e " + str(epochs) +
                               " -v " + str(verbose)).split(),
                              stdout=subprocess.PIPE,
                              env=my_env)
    for line in child1.stdout:
        sys.stdout.write(line)
        import re
        res = re.findall(r'^RES:\s+\(\'(.*?)\',\s+(.*?)\)$', line)
        if len(res) > 0:
            name_ = str(res[0][0])
            score = float(res[0][1])
    child1.communicate()
    queue.put((name_, score))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='run client')
    parser.add_argument('-n', dest='name', type=str)
    parser.add_argument('-e', dest='epochs', type=int)
    parser.add_argument('-v', dest='verbose', type=int)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main_queue = mp.Queue()
    run(queue=main_queue, name=args.name, epochs=args.epochs, verbose=args.verbose)
    d = main_queue.get()
    print  'RES: ', d
