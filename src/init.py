# lazy import ...
print "init"
import matplotlib, sys, os, \
    glob, cPickle, scipy, \
    argparse, errno, json,\
    copy, re,time, imp,datetime
# from operator import add
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os.path as osp
import scipy.io as sio
# import xml.etree.ElementTree as ET
from pprint import pprint
import subprocess
# import cv2, cv
# print "opencv version " + cv2.__version__

_shell_cmd = "rm -f __init__.pyc init.pyc load_transfer_data.pyc  net2net.pyc"
if subprocess.call(_shell_cmd.split()) == 0: print "reload"

keras_backend = "theano"
gpu = True
if keras_backend == "theano":
    _shell_cmd = "cp /home/xlwang/.keras/keras.json.th /home/xlwang/.keras/keras.json"
    if subprocess.call(_shell_cmd.split()) == 0: print "using keras backend theano"
    if not gpu:
        _shell_cmd = "cp /home/xlwang/.theanorc.cpu /home/xlwang/.theanorc"
        if subprocess.call(_shell_cmd.split()) == 0: print "using cpu"
        os.environ['THEANO_FLAGS'] = \
            "floatX=float32,device=cpu," \
            "fastmath=True,ldflags=-lopenblas"
    else:
        _shell_cmd = "cp /home/xlwang/.theanorc.gpu /home/xlwang/.theanorc"
        if subprocess.call(_shell_cmd.split()) == 0: print "using gpu"
else:
    _shell_cmd = "cp /home/xlwang/.keras/keras.json.tf /home/xlwang/.keras/keras.json"
    if subprocess.call(_shell_cmd.split()) == 0: print "using keras backend tf"


# def add_path(path):
#     if path not in sys.path:
#         sys.path.insert(0, path)
# run_root = os.getcwd()
root_dir = osp.normpath(
    osp.join(osp.dirname(__file__),"..")
)

with_caffe = False
if with_caffe == True:
    # if os.path.isfile("/home/luzai/luzai/luzai"):
    #     caffe_root = "/home/luzai/App/caffe"
    # elif os.path.exists("/home/vipa"):
    #     caffe_root = "/home/luzai/App/caffe"
    # else:
    #     caffe_root = input("input your caffe root")
    # os.chdir(caffe_root + "/python")
    # sys.path.insert(0, caffe_root + "/python")
    # print sys.path
    try:
        caffe = imp.load_source('caffe', '/home/luzai/App/caffe/python')
    except Exception as inst:
        print inst.args
        print 'import caffe fail'
    finally:
        pass
        # os.chdir(run_root)

print "\n\n------------------------------\n\n"
