import argparse
import multiprocessing as mp
import os
import subprocess
import time

import numpy as np
import os.path as osp
from Model import IdentityConv, GroupIdentityConv
from Logger import logger

dbg = False
root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)


def run(queue, name, epochs=100, verbose=1, limit_data=False, dataset_type='cifar10'):
    try:
        import keras
        import tensorflow as tf
        from Config import MyConfig
        from Model import MyModel
        from keras.utils.generic_utils import get_custom_objects

        get_custom_objects()['IdentityConv'] = IdentityConv
        get_custom_objects()['GroupIdentityConv'] = GroupIdentityConv

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        with sess.as_default():
            config = MyConfig(name=name, epochs=epochs, verbose=verbose, clean=False, limit_data=limit_data,
                              dataset_type=dataset_type)
            # device = 0
            # with tf.device(device):
            model = MyModel(config=config, model=keras.models.load_model(config.model_path))
            score = model.comp_fit_eval()
            keras.models.save_model(model.model, model.config.model_path)
            queue.put((name, score))
    except Exception as inst:
        print 'INST is'
        print str(inst)
        errors = ['ResourceExhaustedError', 'Resource exhausted: OOM', 'OOM', 'Failed to create session'
                'CUDNN_STATUS_INTERNAL_ERROR', 'Chunk','CUDA_ERROR_OUT_OF_MEMORY']
        for error in errors:
            if error in str(inst):
                queue.put((None, None))
                exit(100)
        exit(200)


def run_self(queue, name, epochs=100, verbose=1, limit_data=False, dataset_type='cifar10'):
    PATH = "/new_disk_1/luzai/App/mpy/bin:"
    os.environ['PATH'] = PATH + os.environ['PATH']
    # subprocess.call("which python".split())
    my_env = os.environ.copy()
    # todo may return : 1. mem not enough 2. model cannot fit

    returncode = 100
    while returncode != 0:
        child1 = subprocess.Popen(("python " + osp.join(root_dir, 'src', 'GAClient.py')
                                   + " -n " + name +
                                   " -e " + str(epochs) +
                                   " -v " + str(verbose) +
                                   " -l " + str(int(limit_data)) +
                                   " -d " + dataset_type).split(),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  env=my_env)
        # subprocess.call(("python " + osp.join(root_dir, 'src', 'GAClient.py')
        #                            + " -n " + name +
        #                            " -e " + str(epochs) +
        #                            " -v " + str(verbose)).split())
        returncode = child1.wait()
        logger.info('returncode is {}'.format(returncode))
        stdout, stderr = child1.communicate()
        if returncode == 100 or returncode == -6:
            logger.info('No enough mem, model {} wait for 5min'.format(name))
            if dbg:
                time.sleep(10)
            else:
                time.sleep(np.random.choice([10, 30, 60]))
            logger.info('5 min passed, {} Continue try'.format(name))
        elif returncode == 0:
            logger.info('model {} Fit Success'.format(name))
            break
        else:
            logger.error('UNKNOWN except stderr is {1} stdout is {0} Maybe just not enough mem, have another try'.format(stdout, stderr))
    print stdout  # , stderr
    import re
    name_ = re.findall(r'NAME:\s+(.*)', stdout)[0]
    score = float(re.findall(r'SCORE:\s+(.*)', stdout)[0])
    assert 'name_' in locals(), '{}{}'.format(returncode, name)

    queue.put((name_, score))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='run client')
    parser.add_argument('-n', dest='name', type=str)
    parser.add_argument('-e', dest='epochs', type=int)
    parser.add_argument('-v', dest='verbose', type=int)
    parser.add_argument('-l', dest='limit_data', type=int)
    parser.add_argument('-d', dest='dataset_type', type=str)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main_queue = mp.Queue()
    with open('gaclient', 'w') as f:
        f.write(str(args))

    run(queue=main_queue, name=args.name, epochs=args.epochs, verbose=args.verbose, limit_data=args.limit_data,
        dataset_type=args.dataset_type)
    d = main_queue.get()
    print 'NAME: ', d[0]
    print 'SCORE: {:.13f}'.format(d[1])
    exit(0)
