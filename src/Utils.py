# TODO add vis util (ipython/python)
import  subprocess,os,json
import os.path as osp
from Init import root_dir
from keras.utils import vis_utils
from Model import  MyModel

# TODO for python and ipython vis
# TODO tf function like summary in callback, grap_def to dot, get output_tensor
# for python
# TODO map length name to clean ones
def vis_model(model, name='net2net'):
    if  isinstance(model,MyModel):
        model=model.model
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
    vis_utils.plot_model(model, to_file=name + '.pdf', show_shapes=True)
    vis_utils.plot_model(model, to_file=name + '.png', show_shapes=True)
    os.chdir("../../src")

def vis_graph(graph,name='net2net'):
    _shell_cmd = "mkdir -p " + osp.join(root_dir, "output", name)
    subprocess.call(_shell_cmd.split())
    os.chdir(osp.join(root_dir, "output", name))
    with open(name + "_graph.json", "w") as f:
        f.write(graph.to_json())
    import networkx as nx
    nx.draw(graph, with_labels=True)
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.savefig('graph.png')
    os.chdir("../../src")
