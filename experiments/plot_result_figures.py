# -*- coding: UTF-8 -*-

def plot_prob_curve_during_evolution():
    # 画出5种操作的概率曲线随着进化次数的变化情况
    import numpy as np
    import matplotlib.pyplot as plt

    choice_weight_list = np.load('../output/choice_prob.npy')
    choice = np.load('../output/choice_idx.npy')
    # print choice
    print choice_weight_list

    iter = np.array([i for i in range(len(choice_weight_list))])
    weight_dp = []
    weight_deeper = []
    weight_wider = []
    weight_add_skip = []
    weight_add_group = []
    for weight in choice_weight_list:
        weight_dp.append(weight['deeper_with_pooling'])
        weight_deeper.append(weight['deeper'])
        weight_wider.append(weight['wider'])
        weight_add_skip.append(weight['add_skip'])
        weight_add_group.append(weight['add_group'])

    weight_dp = np.array(weight_dp)
    weight_deeper = np.array(weight_deeper)
    weight_wider = np.array(weight_wider)
    weight_add_skip = np.array(weight_add_skip)
    weight_add_group = np.array(weight_add_group)

    sum = weight_dp + weight_deeper + weight_wider + weight_add_skip + weight_add_group
    weight_dp = 1.0 * weight_dp / sum
    weight_deeper = 1.0 * weight_deeper / sum - 0.005
    weight_wider = 1.0 * weight_wider / sum + 0.005
    weight_add_skip = 1.0 * weight_add_skip / sum - 0.005
    weight_add_group = 1.0 * weight_add_group / sum + 0.005

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    # plot prob line
    ax.plot(iter, weight_dp, linewidth=2.5, label="deeper_with_pooling")
    ax.plot(iter, weight_deeper, linewidth=2.5, label="deeper")
    ax.plot(iter, weight_wider, linewidth=2.5, label="wider")
    ax.plot(iter, weight_add_skip, linewidth=2.5, label="add_skip")
    ax.plot(iter, weight_add_group, linewidth=2.5, label="add_group")

    # plot choosen list
    choice_point_x = []
    choice_point_y = []
    for idx, item in enumerate(choice):
        choice_point_x.append(idx)
        if item == 'deeper_with_pooling':
            choice_point_y.append(weight_dp[idx])
        elif item == 'deeper':
            choice_point_y.append(weight_deeper[idx])
        elif item == 'wider':
            choice_point_y.append(weight_wider[idx])
        elif item == 'add_skip':
            choice_point_y.append(weight_add_skip[idx])
        elif item == 'add_group':
            choice_point_y.append(weight_add_group[idx])
    plt.plot(choice_point_x, choice_point_y, 'yo', label="selected op.")

    lgd = ax.legend(loc='center right', bbox_to_anchor=(1.5, 0.5))
    ax.set_title('chosen probility of 5 operations during evolition')
    ax.set_xlabel('evolution times')
    ax.set_ylabel('chosen probility')

    ax.set_xlim([-1, len(choice_point_x)])
    ax.set_ylim([-0.02, 0.5])

    fig.savefig('chosen_probility.png', dpi=300, format='png', bbox_extra_artists=(lgd,), bbox_inches='tight')


def plot_evolution_result():
    # 根据gl_checkpoint.csv画出树，确认父子关系
    import csv
    import re
    import pydot  # import pydot or you're not going to get anywhere my friend :D
    import os
    #os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz2.38\\bin\\'

    # 1.根据gl_checkpoint.csv文件读取所有的边的信息，保存成一个字典
    parent_child_dict = {}
    reader = csv.reader(open("../output/gl_checkpoint.csv"))
    for parent, child, op in reader:
        if parent in parent_child_dict.keys():
            parent_child_dict[parent].append(child)
        else:
            parent_child_dict[parent] = [child]

    print parent_child_dict

    # 2.遍历这个字典，删除多余的same节点
    for (parent, childs) in parent_child_dict.items():
        # 遍历每一个孩子
        for child in childs:
            iter, ind, op = re.findall('ga_iter_(\d+)_ind_(\d+)(?:_(\w+))*', child)[0]
            # 如果这个child op是same，遍历这个child的所有孩子节点
            if op == 'same' and child in parent_child_dict.keys():
                for child_child in parent_child_dict[child]:
                    iter2, ind2, op2 = re.findall('ga_iter_(\d+)_ind_(\d+)(?:_(\w+))*', child_child)[0]
                    if iter2 == iter and op2 != 'same':  # 需要移除这个节点
                        # parent_child_dict[parent].remove(child)
                        parent_child_dict[child].remove(child_child)
                        parent_child_dict[parent].append(child_child)

    # create a new graph
    graph = pydot.Dot(graph_type='graph')

    for (parent, childs) in parent_child_dict.items():
        for child in childs:
            iter, ind, op = re.findall('ga_iter_(\d+)_ind_(\d+)(?:_(\w+))*', child)[0]
            if op == 'deeper_with_pooling':
                color = 'blue'
            elif op == 'deeper':
                color = 'yellow'
            elif op == 'wider':
                color = 'red'
            elif op == 'add_skip':
                color = 'pink'
            elif op == 'add_group':
                color = 'orange'
            else:
                color = 'black'
            edge = pydot.Edge(parent, child, color=color, )
            graph.add_edge(edge)

    graph.write_png('relationship_reduced.png')


    # 根据前面简化过的 parent_child_dict 画出最后的结果图
    import json
    import glob
    import re
    import os
    import csv
    import numpy as np
    import matplotlib.pyplot as plt

    model_x = []
    model_y = []

    fig = plt.figure(1)
    ax = fig.add_subplot(111)

    # 连通分量，不同的连通分量用不同的标记
    markers = ['o', 'v', 's', '*', 'h', 'd', '+', 'x']

    import networkx as nx
    G = nx.Graph()

    for (parent, childs) in parent_child_dict.items():
        for child in childs:
            G.add_edge(parent, child)
    comps = nx.connected_components(G)
    # for comp in comps:
    #    print(comp)


    for (parent, childs) in parent_child_dict.items():
        for child in childs:
            # 读取parent 和 child 对应的文件夹，把数据读出来
            parent_path = "..\\output\\" + parent + "\\info.json"
            iter, ind, op = re.findall('ga_iter_(\d+)_ind_(\d+)(?:_(\w+))*', parent)[0]
            if os.path.isfile(parent_path) == True:
                with open(parent_path, 'r') as f:
                    data = json.load(f)
                    parent_point = [iter, data['param']]

            child_path = "..\\output\\" + child + "\\info.json"
            iter, ind, op = re.findall('ga_iter_(\d+)_ind_(\d+)(?:_(\w+))*', child)[0]
            if os.path.isfile(child_path) == True:
                with open(child_path, 'r') as f:
                    data = json.load(f)
                    child_point = [iter, data['param']]

            # choose color
            if op == 'deeper_with_pooling':
                color = 'blue'
            elif op == 'deeper':
                color = 'yellow'
            elif op == 'wider':
                color = 'red'
            elif op == 'add_skip':
                color = 'pink'
            elif op == 'add_group':
                color = 'orange'
            else:
                color = 'black'

                # draw line
            ax.plot([parent_point[0], child_point[0]],\
                    [parent_point[1], child_point[1]], linestyle='-', color=color)

            # draw point
            print child
            for idx, comp in enumerate(comps):
                if child in comp:
                    print (child, idx)
                    ax.plot(child_point[0], child_point[1], markers[idx])
                    break

    # add info
    ax.set_title('param size change during evolution')
    ax.set_xlabel('evolution times')
    ax.set_ylabel('params size (MB)')

    fig.savefig('evolution_his.png', dpi=300, format='png')

plot_evolution_result()