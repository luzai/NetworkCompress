import os
import re
import subprocess

import keras, json
import keras.backend as K
import matplotlib
import numpy as np

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import os.path as osp
from Logger import logger

import random
from IPython import embed

root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)

from IPython.display import display, HTML, SVG
import csv


def line_append(line, file_path):
    # csv fields is ['before', 'after', 'operation']
    with open(file_path, 'a') as f:
        writer = csv.writer(f)
        writer.writerow(line)

def read_json(file_path):
    with open(file_path, 'r') as f:
        obj = json.load(f)
    return obj


def write_json(obj, file_path):
    with open(file_path, 'w') as f:
        json.dump(obj, f, indent=4, separators=(',', ': '))


def choice_dict(mdict, size):
    # for test
    #choice = np.random.choice(mdict.keys(), size=size, replace=False)
    #return {name: model for name, model in mdict.items() if name in choice}

    import Queue
    queue = Queue.PriorityQueue()
    for name, model in mdict.items():
        queue.put((-model.score, name))
    res = {}
    for i in range(size):
        _, name = queue.get()
        res[name] = mdict[name]
    return res


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


# descrapted
def dict2list(mdict):
    return [(a, b) for a, b in mdict.items()]


def list2dict(mlist):
    return {a: b for (a, b) in mlist}


def mkdir_p(name, delete=True):
    # it is not good for debug, but non-delete will also cause resource race
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if delete and tf.gfile.Exists(name):
        tf.gfile.DeleteRecursively(name)
        logger.warning('delte files under {}'.format(name))
    tf.gfile.MakeDirs(name)


def i_vis_model(model):
    from keras.utils import vis_utils
    SVG(vis_utils.model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


def vis_model(model, name='net2net', show_shapes=True):
    from keras.utils import vis_utils
    path = osp.dirname(name)
    name = osp.basename(name)
    if path == '':
        path = name
    sav_path = osp.join(root_dir, "output", path)
    mkdir_p(sav_path, delete=False)
    keras.models.save_model(model, osp.join(sav_path, name + '.h5'))
    try:
        # vis_utils.plot_model(model, to_file=osp.join(sav_path, name + '.pdf'), show_shapes=show_shapes)
        vis_utils.plot_model(model, to_file=osp.join(sav_path, name + '.png'), show_shapes=show_shapes)
    except Exception as inst:
        logger.error("cannot keras.plot_model {}".format(inst))


def vis_graph(graph, name='net2net', show=False):
    path = osp.dirname(name)
    name = osp.basename(name)
    if path == '':
        path = name
    mkdir_p(osp.join(root_dir, "output", path), delete=False)
    restore_path = os.getcwd()
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
    os.chdir(restore_path)


def nvidia_smi():
    # todo now we only use gpu 0
    proc = subprocess.Popen("nvidia-smi --query-gpu=index,memory.free --format=csv".split()
                            , stdout=subprocess.PIPE)
    (out, err) = proc.communicate()
    free = re.findall(r'0,\s+(\d+)\s+MiB', out)
    free = [int(val) for val in free]
    proc = subprocess.Popen("nvidia-smi --query-gpu=index,memory.total --format=csv".split()
                            , stdout=subprocess.PIPE)
    (out, err) = proc.communicate()
    ttl = re.findall(r'0,\s+(\d+)\s+MiB', out)
    ttl = [int(val) for val in ttl]

    ratio = float(max(free)) / max(ttl)
    return ratio


def count_weight(model):
    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)]) * 4. / 1024. / 1024.
    # convert to MB
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]) * 4. / 1024. / 1024.

    return trainable_count, non_trainable_count


def print_graph_info():
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
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
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
    import tensorflow as tf
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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


def add_indent(str):
    import re
    return re.sub('\n', '\n\t\t', str)


def to_single_dir():
    restore_path = os.getcwd()
    os.chdir(root_dir)
    for parent, dirnames, filenames in os.walk('output/tf_tmp'):
        filenames = sorted(filenames)
        if len(filenames) == 1:
            continue
        for ind, fn in enumerate(filenames):
            subprocess.call(('mkdir -p ' + parent + '/' + str(ind)).split())
            subprocess.call(('mv ' + parent + '/' + fn + ' ' + parent + '/' + str(ind) + '/').split())
        print parent, filenames
    os.chdir(restore_path)


def load_data_svhn():
    import scipy.io as sio
    import os.path
    import commands

    if os.path.isdir(osp.join(root_dir, 'data/SVHN')) == False:
        os.mkdir(osp.join(root_dir, 'data/SVHN'))

    data_set = []
    if os.path.isfile(osp.join(root_dir, 'data/SVHN/train_32x32.mat')) == False:
        data_set.append("train")
    if os.path.isfile(osp.join(root_dir, 'data/SVHN/test_32x32.mat')) == False:
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
        print ('download SVHN ' + set + ' data, Please wait.')
        url = "http://ufldl.stanford.edu/housenumbers/" + set + "_32x32.mat"
        response = requests.get(url, stream=True)
        with open("data/SVHN/" + set + "_32x32.mat", "wb") as handle:
            for data in tqdm(response.iter_content()):
                handle.write(data)

    train_data = sio.loadmat(root_dir + '/data/SVHN/train_32x32.mat')
    train_x = train_data['X']
    train_y = train_data['y']

    test_data = sio.loadmat(root_dir + '/data/SVHN/test_32x32.mat')
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
