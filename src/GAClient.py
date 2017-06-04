import multiprocessing as mp
import argparse, os, sys
import subprocess, time
import os.path as osp
import numpy as np

root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)


def run(queue, name, epochs=100, verbose=1,limit_data=False):
    try:
        import keras
        import tensorflow as tf

        from Config import MyConfig
        from Model import MyModel

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        with sess.as_default():
            config = MyConfig(name=name, epochs=epochs, verbose=verbose, clean=False,limit_data=limit_data)
            # device = 0
            # with tf.device(device):
            # try:
            model = MyModel(config=config, model=keras.models.load_model(config.model_path))
            print "start train {}".format(name)
            score = model.comp_fit_eval()
            queue.put((name, score))
    except Exception as inst:
        print inst
        with open('err', 'w')as f:
            f.write(str(inst))
        if 'ResourceExhaustedError' in str(inst):
            queue.put((None, None))
            exit(-2)
        else:
            queue.put((None, None))
            exit(-2)


def run_self(queue, name, epochs=100, verbose=1,limit_data=False):
    my_env = os.environ.copy()
    # todo may return : 1. mem not enough 2. model cannot fit
    PATH = "/new_disk_1/luzai/App/mpy/bin:$PATH"
    returncode = 254
    while returncode !=0:
        child1 = subprocess.Popen(("python " + osp.join(root_dir, 'src', 'GAClient.py')
                                   + " -n " + name +
                                   " -e " + str(epochs) +
                                   " -v " + str(verbose)+
                                  " -l "+str(int(limit_data))).split(),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  env=my_env)
        # subprocess.call(("python " + osp.join(root_dir, 'src', 'GAClient.py')
        #                            + " -n " + name +
        #                            " -e " + str(epochs) +
        #                            " -v " + str(verbose)).split())
        returncode = child1.wait()
        print 'returncode is {}'.format(returncode)

        stdout, stderr = child1.communicate()
        if returncode !=0:
            # -6 CUDNN_STATUS_INTERNAL_ERROR
            # print stdout,stderr
            print 'No enough mem, model {} wait for 5min'.format(name)
            time.sleep(np.random.choice([10,300,600]))
            print '5 min passed, {} Continue try'.format(name)

    print stdout # , stderr
    import re
    name_=re.findall(r'NAME:\s+(.*)',stdout)[0]
    score=float(re.findall(r'SCORE:\s+(.*)',stdout)[0])
    assert 'name_' in locals(), '{}{}'.format(returncode, name)

    queue.put((name_, score))


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='run client')
    parser.add_argument('-n', dest='name', type=str)
    parser.add_argument('-e', dest='epochs', type=int)
    parser.add_argument('-v', dest='verbose', type=int)
    parser.add_argument('-l',dest='limit_data',type=int)
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    main_queue = mp.Queue()
    with open('gaclient','w') as f :
        f.write(str(args))

    run(queue=main_queue, name=args.name, epochs=args.epochs, verbose=args.verbose,limit_data=args.limit_data)
    d = main_queue.get()
    print 'NAME: ', d[0]
    print 'SCORE: {:.13f}'.format(d[1])
    exit(0)
