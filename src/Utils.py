# TODO add vis util (ipython/python)
import json
import subprocess

import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import os
import os.path as osp
import tensorflow as tf
from keras.utils import vis_utils

# from Init import root_dir
root_dir = osp.normpath(
    osp.join(osp.dirname(__file__), "..")
)
from Model import MyModel

from IPython.display import display, HTML, SVG


# TODO tf function like summary in callback, grap_def to dot, get output_tensor
# TODO map length name to clean ones

def i_vis_model(model):
    SVG(vis_utils.model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))


def vis_model(model, name='net2net'):
    if isinstance(model, MyModel):
        model = model.model
    _shell_cmd = "mkdir -p " + osp.join(root_dir, "output", name)
    subprocess.call(_shell_cmd.split())
    os.chdir(osp.join(root_dir, "output", name))
    model.save_weights(name + ".h5")
    with open(name + "_model.json", "w") as f:
        json.dump(
            json.loads(model.to_json()),
            f,
            indent=2
        )
    try:
        vis_utils.plot_model(model, to_file=name + '.pdf', show_shapes=True)
        vis_utils.plot_model(model, to_file=name + '.png', show_shapes=True)
    except Exception as inst:
        print inst
    os.chdir("../../src")


def vis_graph(graph, name='net2net'):
    _shell_cmd = "mkdir -p " + osp.join(root_dir, "output", name)
    subprocess.call(_shell_cmd.split())
    os.chdir(osp.join(root_dir, "output", name))
    with open(name + "_graph.json", "w") as f:
        f.write(graph.to_json())

    plt.close('all')
    nx.draw(graph, with_labels=True)
    try:
        plt.show()
    except Exception as inst:
        print inst
    plt.savefig('graph.png')
    os.chdir("../../src")


def nvidia_smi():
    proc = subprocess.Popen(["nvidia-smi"], stdout=subprocess.PIPE, shell=True)
    (out, err) = proc.communicate()
    print "program output:", out


import numpy as np
import keras.backend as K


def count_weight(model):
    # TODO way 1 similar to above
    # Way2

    if isinstance(model, MyModel):
        model = model.model
    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])

    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

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


# also test file
if __name__ == "__main__":
    plt.plot([1, 2, 4], [3, 6, 10])
    plt.show()
