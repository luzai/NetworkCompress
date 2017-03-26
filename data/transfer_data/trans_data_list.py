# -*- coding:utf-8 -*-

import os
import os.path
rootdir = '.'

file = open('trans_data_list.txt', 'w')

cate_list = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'trunk']

for parent, dirnames, filenames in os.walk(rootdir):
    for filename in filenames:                        #输出文件信息
        if parent != '.' and parent != '..':
            label = cate_list.index(parent[2:])
            line = os.path.join(parent[2:],filename) + " " + str(label) + '\n'
            print(line)
            file.write(line)

file.close()     
