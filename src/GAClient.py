import argparse
import multiprocessing as mp
import os
import subprocess
import time, Utils

import numpy as np
import os.path as osp
from Model import IdentityConv, GroupIdentityConv
from Logger import logger

root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)
NOMEM = 100


class Client:
    PATH = "/new_disk_1/luzai/App/mpy/bin:/home/vipa-spark/luzai/anaconda/bin:"
    os.environ['PATH'] = PATH + os.environ['PATH']
    my_env = os.environ.copy()

    def __init__(self):
        self.max_run = 6
        self.run = 0
        self.tasks = {}
        self.kwargs = {}
        self.scores = {}

    def run_self(self, kwarg):
        name, epochs, verbose, limit_data, dataset_type = kwarg['name'], kwarg['epochs'], kwarg['verbose'], kwarg[
            'limit_data'], kwarg['dataset_type']
        self.kwargs[name] = kwarg
        logger.debug('deploy the task with kwarg {}'.format(kwarg))
        self.tasks[name] = subprocess.Popen(("python " + osp.join(root_dir, 'src', 'GAClient.py')
                                             + " -n " + name +
                                             " -e " + str(epochs) +
                                             " -v " + str(verbose) +
                                             " -l " + str(int(limit_data)) +
                                             " -d " + dataset_type).split(),
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                            env=Client.my_env)
        time.sleep(5)
        self.run += 1

    def wait(self):
        for name, task in self.tasks.items():
            returncode = task.wait()
            stdout, stderr = task.communicate()
            stdout = Utils.add_indent(stdout)
            stderr = Utils.add_indent(stderr)
            if returncode == 0:
                name, score = self.parse_score(stdout)
                self.scores[name] = score
                logger.info('model {} Fit success with score {}'.format(name, score))
                logger.info('success stdout is {}'.format(stdout))
                self.run -= 1
            else:
                while returncode != 0:
                    # self.max_run -= 1
                    if Utils.nvidia_smi() < .2:
                        time.sleep(10)
                    logger.info('fit fail return code is {}'.format(returncode))
                    if returncode != NOMEM:
                        logger.error(
                            'some unknown error maybe no enough mem' +
                            'stderr ==>  {1}, stdout ==> {0}'.format(stdout,
                                                                     stderr))

                    kwarg = self.kwargs[name]
                    self.run_self(kwarg)
                    returncode = self.tasks[name].wait()
                stdout, stderr = self.tasks[name].communicate()
                stdout = Utils.add_indent(stdout)
                stderr = Utils.add_indent(stderr)
                name, score = self.parse_score(stdout)
                self.scores[name] = score
                logger.info('model {} Fit success with score {}'.format(name, score))
                logger.info('success stdout is {}'.format(stdout))
                self.run -= 1

    @staticmethod
    def parse_score(stdout):
        import re
        name_ = re.findall(r'NAME:\s+(.*)', stdout)[0]
        score = float(re.findall(r'SCORE:\s+(.*)', stdout)[0])
        return name_, score


def run(name, epochs=100, verbose=1, limit_data=False, dataset_type='cifar10'):
    try:
        import keras
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
        import tensorflow as tf
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
            logger.debug('model {} start fit epochs {}'.format(name, epochs))
            score = model.comp_fit_eval()
            keras.models.save_model(model.model, model.config.model_path)
            return name, score

    except Exception as inst:
        print 'INST is'
        print str(inst)
        errors = ['ResourceExhaustedError',
                  'Resource exhausted: OOM',
                  'OOM', 'Failed to create session',
                  'CUDNN_STATUS_INTERNAL_ERROR', 'Chunk',
                  'CUDA_ERROR_OUT_OF_MEMORY']
        for error in errors:
            if error in str(inst):
                return NOMEM, NOMEM
        return None, None


if __name__ == '__main__':
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


    args = parse_args()
    name, score = run(name=args.name, epochs=args.epochs, verbose=args.verbose, limit_data=args.limit_data,
                      dataset_type=args.dataset_type)
    if name is None:
        exit(200)
    elif name == NOMEM:
        exit(NOMEM)
    else:
        print 'NAME: ', name
        print 'SCORE: {:.13f}'.format(score)
        exit(0)
