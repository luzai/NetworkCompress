import os
import re
import subprocess

import keras
import keras.backend as K
import matplotlib
import numpy as np
import tensorflow as tf
import pdb
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import os.path as osp
from Logger import logger
from keras.utils import vis_utils
import random
from IPython import embed

# from Init import root_dir
root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)

from IPython.display import display, HTML, SVG


# TODO map lengthy name to clean name

def choice_dict(mdict, size):
    choice = np.random.choice(mdict.keys(), size=size, replace=False)
    return {name: model for name, model in mdict.items() if name in choice}


def choice_dict_keep_latest(mdict, size):
    # find the max ind model
    max_ind = -1
    for name, model in mdict.items():
        # iter, ind = filter(str.isdigit, name)
        iter, ind = re.findall('ga_iter_(\d+)_ind_(\d+)', name)[0]
        # logger.debug('iter {} ind {} max_ind {}'.format(iter, ind, max_ind))
        if int(ind) > max_ind:
            max_ind = int(ind)
            latest = {name: model}
    assert 'latest' in locals().keys()
    return latest

def weight_choice(list, weight_dict):
    weight = []
    for element in list:
        weight.append(weight_dict[element])

    weight = np.array(weight).astype('float')
    weight = weight / weight.sum()
    return int(np.random.choice(range(len(weight)), p=weight))
    # t = random.randint(0, sum(weight) - 1)
    # for i, val in enumerate(weight):
    #     t -= val
    #     if t < 0:
    #         return i


# descrapted
def dict2list(mdict):
    return [(a, b) for a, b in mdict.items()]


def list2dict(mlist):
    return {a: b for (a, b) in mlist}


def mkdir_p(name, delete=True):
    # TODO it is not good for debug, but non-delete will also cause resource race
    if delete and tf.gfile.Exists(name):
        tf.gfile.DeleteRecursively(name)
    tf.gfile.MakeDirs(name)


def i_vis_model(model):
    SVG(vis_utils.model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


def vis_model(model, name='net2net', show_shapes=True):
    path = osp.dirname(name)
    name = osp.basename(name)
    if path == '':
        path = name
    mkdir_p(osp.join(root_dir, "output", path), delete=False)
    os.chdir(osp.join(root_dir, "output", path))
    keras.models.save_model(model, name + '.h5')
    try:
        vis_utils.plot_model(model, to_file=name + '.pdf', show_shapes=show_shapes)
        vis_utils.plot_model(model, to_file=name + '.png', show_shapes=show_shapes)
    except Exception as inst:
        logger.error("cannot keras.plot_model {}".format(inst))
    os.chdir(osp.join(root_dir, 'src'))


def vis_graph(graph, name='net2net', show=False):
    path = osp.dirname(name)
    name = osp.basename(name)
    if path == '':
        path = name
    mkdir_p(osp.join(root_dir, "output", path), delete=False)
    os.chdir(osp.join(root_dir, "output", path))
    with open(name + "_graph.json", "w") as f:
        f.write(graph.to_json())
    try:
        plt.close('all')
        nx.draw(graph, with_labels=True)
        if show:
            plt.show()
        plt.savefig('graph.png')
        # plt.close('all')
    except Exception as inst:
        logger.warning(inst)
    os.chdir(osp.join(root_dir, 'src'))


def nvidia_smi():
    proc = subprocess.Popen("nvidia-smi --query-gpu=index,memory.free --format=csv".split()
                            , stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    res = re.findall(r'\s+(\d+)MiB', out)
    res = [int(val) for val in res]
    return res


def count_weight(model):
    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)]) * 4. / 1024. / 1024.
    # convert to MB
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]) * 4. / 1024. / 1024.

    return trainable_count, non_trainable_count


def print_graph_info():
    graph = tf.get_default_graph()
    graph.get_tensor_by_name("Placeholder:0")
    layers = [op.name for op in graph.get_operations() if op.type == "Placeholder"]
    print [graph.get_tensor_by_name(layer + ":0") for layer in layers]
    print [op.type for op in graph.get_operations()]
    print [n.name for n in tf.get_default_graph().as_graph_def().node]
    print [v.name for v in tf.global_variables()]
    print graph.get_operations()[20]


def i_vis_graph(graph_def, max_const_size=32):
    """Visualize TensorFlow graph."""
    if hasattr(graph_def, 'as_graph_def'):
        graph_def = graph_def.as_graph_def()
    strip_def = strip_consts(graph_def, max_const_size=max_const_size)
    code = """
        <script>
          function load() {{
            document.getElementById("{id}").pbtxt = {data};
          }}
        </script>
        <link rel="import" href="https://tensorboard.appspot.com/tf-graph-basic.build.html" onload=load()>
        <div style="height:600px">
          <tf-graph-basic id="{id}"></tf-graph-basic>
        </div>
    """.format(data=repr(str(strip_def)), id='graph' + str(np.random.rand()))

    iframe = """
        <iframe seamless style="width:800px;height:620px;border:0" srcdoc="{}"></iframe>
    """.format(code.replace('"', '&quot;'))
    display(HTML(iframe))


def strip_consts(graph_def, max_const_size=32):
    """Strip large constant values from graph_def."""
    strip_def = tf.GraphDef()
    for n0 in graph_def.node:
        n = strip_def.node.add()
        n.MergeFrom(n0)
        if n.op == 'Const':
            tensor = n.attr['value'].tensor
            size = len(tensor.tensor_content)
            if size > max_const_size:
                tensor.tensor_content = tf.compat.as_bytes("<stripped %d bytes>" % size)
    return strip_def


def to_single_dir():
    os.chdir(root_dir)
    for parent, dirnames, filenames in os.walk('output/tf_tmp'):
        filenames = sorted(filenames)
        if len(filenames)==1:
            continue
        for ind, fn in enumerate(filenames):
            subprocess.call(('mkdir -p ' + parent + '/' + str(ind)).split())
            subprocess.call(('mv ' + parent + '/' + fn + ' ' + parent + '/' + str(ind) + '/').split())
        print parent, filenames


def load_data_svhn():
    import scipy.io as sio
    import os.path
    import commands

    if os.path.isdir('../data/SVHN') == False:
        os.mkdir('../data/SVHN')

    data_set = []
    if os.path.isfile('../data/SVHN/train_32x32.mat') == False:
        data_set.append("train")
    if os.path.isfile('../data/SVHN/test_32x32.mat') == False:
        data_set.append("test")

    try:
        import requests
        from tqdm import tqdm
    except:
        # use pip to install these packages:
        # pip install tqdm
        # pip install requests
        print('please install requests and tqdm package first.')

    for set in data_set:
        print ('download SVHN ' + set + ' data, Please wait.' )
        url = "http://ufldl.stanford.edu/housenumbers/" + set + "_32x32.mat"
        response = requests.get(url, stream=True)
        with open("../data/SVHN/" + set + "_32x32.mat", "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)

    train_data = sio.loadmat('../data/SVHN/train_32x32.mat')
    train_x = train_data['X']
    train_y = train_data['y']

    test_data = sio.loadmat('../data/SVHN/test_32x32.mat')
    test_x = test_data['X']
    test_y = test_data['y']

    # 1 - 10 to 0 - 9
    train_y = train_y - 1
    test_y = test_y - 1

    train_x = np.transpose(train_x, (3, 0, 1, 2))
    test_x = np.transpose(test_x, (3, 0, 1, 2))

    return (train_x, train_y), (test_x, test_y)


if __name__ == "__main__":
    to_single_dir()
