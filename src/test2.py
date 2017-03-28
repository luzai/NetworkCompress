# from init import *
import matplotlib, sys, os, \
    glob, cPickle, scipy, \
    argparse, errno, json,\
    copy, re,time, imp,datetime
# from operator import add
matplotlib.use("TkAgg")
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
saves=[]
save=[]
with open("./vgg_net2net.log3",'r') as f:
    line_sav=f.readlines()

    for line in line_sav:

        if "Attention" in line:
            if save != []:
                saves.append(save)
            save=[]
        p = re.compile(r'\d+s - loss: \d+.\d+ - acc: \d+.\d+ - val_loss: \d+.\d+ - val_acc: (\d+.\d+)')
        t=p.search(line)
        if t is not None:
            save.append(float(t.group(1)))
            # print save
# import matplotlib.pyplot as p
# p.switch_backend('TkAgg')
# print saves
# print saves[-1]
# print len(saves)
acc_all=[0]
legend=[]
fig=plt.figure(figsize=(100,5),facecolor=(1,1,1))
plt.hold("on")
for ind,save in enumerate(saves):
    legend+=["stage"+str(ind+1)]
    plt.plot(np.arange(start=len(acc_all)-1, stop=len(acc_all + save)), np.array([acc_all[-1]]+save))
    acc_all+=save

#     print acc_all
#     print len(acc_all)
# print(len(acc_all))
plt.legend(legend)
plt.show()