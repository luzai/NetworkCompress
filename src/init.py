# lazy import ...
print "init"
import matplotlib, sys, os, \
    glob, cPickle, scipy, \
    argparse, errno, json,\
    copy, re,time
from operator import add

import pandas as pd
import numpy as np
import os.path as osp
import scipy.io as sio
# import xml.etree.ElementTree as ET
from pprint import pprint
import subprocess
# import cv2, cv
# print "opencv version " + cv2.__version__

cmd="rm -f __init__.pyc init.pyc load_transfer_data.pyc  net2net.pyc"
if subprocess.call(cmd.split())==0 : print "reload"

keras_backend="theano"
gpu=True
if keras_backend=="theano":
    cmd="cp /home/xlwang/.keras/keras.json.th /home/xlwang/.keras/keras.json"
    if subprocess.call(cmd.split())==0: print "using keras backend theano"
    if not gpu:
        cmd = "cp /home/xlwang/.theanorc.cpu /home/xlwang/.theanorc"
        if subprocess.call(cmd.split())==0: print "using cpu"
        os.environ['THEANO_FLAGS'] = \
            "floatX=float32,device=cpu," \
            "fastmath=True,ldflags=-lopenblas"
        import theano
    else:
        cmd = "cp /home/xlwang/.theanorc.gpu /home/xlwang/.theanorc"
        if subprocess.call(cmd.split())==0: print "using gpu"
        import theano
else:
    cmd="cp /home/xlwang/.keras/keras.json.tf /home/xlwang/.keras/keras.json"
    if subprocess.call(cmd.split())==0 : print "using keras backend tf"
    import tensorflow as tf
import keras

import seaborn as sns
import matplotlib.pyplot as plt

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)
with_caffe=False
if with_caffe==True:
    run_root = os.getcwd()
    file_root = osp.dirname(__file__)
    if os.path.isfile("/home/luzai/luzai/luzai"):
        caffe_root = "/home/luzai/App/caffe"
    elif os.path.exists("/home/vipa"):
        caffe_root = "/home/luzai/App/caffe"
    else:
        caffe_root=input("input your caffe root")
    os.chdir(caffe_root + "/python")
    sys.path.insert(0, caffe_root + "/python")
    print sys.path

    try:
        add_path(caffe_root + "/python")
        import caffe
    except Exception as inst:
        print inst.args
        print 'import caffe fail'
    finally:
        os.chdir(run_root)

print "\n\n------------------------------\n\n"
