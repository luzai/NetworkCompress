from net2net import *
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
acc_all=[0]
legend=[]
fig=plt.figure(figsize=(100,5),facecolor=(1,1,1))
plt.hold("on")
for ind,save in enumerate(saves):
    legend+=["stage"+str(ind+1)]
    plt.plot(np.arange(start=len(acc_all)-1, stop=len(acc_all + save)), np.array([acc_all[-1]]+save))
    acc_all+=save
plt.legend(legend)
# plt.show()
plt.savefig("parsed_lod.png")