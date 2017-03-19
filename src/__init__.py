# lazy import ...

import matplotlib, sys, os, glob, cPickle, scipy, argparse, errno, json,copy, re,keras
from operator import add
# import tensorflow as tf
import pandas as pd
import numpy as np
import os.path as osp
import scipy.io as sio
# import xml.etree.ElementTree as ET
from pprint import pprint
import subprocess
from subprocess import call
# import cv2, cv
# print "opencv version " + cv2.__version__
run_root = os.getcwd()
file_root = osp.dirname(__file__)

import seaborn as sns
# if os.path.isfile("/home/luzai/luzai/luzai"):
#     caffe_root = "/home/luzai/App/caffe"
# elif os.path.exists("/home/vipa"):
#     caffe_root = "/home/luzai/App/caffe"
# else:
#     caffe_root=input("input your caffe root")
import matplotlib.pyplot as plt

# os.chdir(caffe_root + "/python")


# sys.path.insert(0, caffe_root + "/python")
# print sys.path

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


# try:
#     add_path(caffe_root + "/python")
#     import caffe
# except Exception as inst:
#     print inst.args
#     print 'import caffe fail'
# finally:
#     os.chdir(run_root)

print "\n\n------------------------------\n\n"