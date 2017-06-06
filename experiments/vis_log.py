from Net2Net import *

saves = []
save = []
with open("./vgg_net2net.log2", 'r') as f:
    line_sav = f.readlines()

    for line in line_sav:

        if "Attention" in line:
            if save != []:
                saves.append(save)
            save = []
        p = re.compile(r'\d+s - loss: \d+.\d+ - acc: \d+.\d+ - val_loss: \d+.\d+ - val_acc: (\d+.\d+)')
        t = p.search(line)
        if t is not None:
            save.append(float(t.group(1)))

acc_all = [0]
legend = []
fig = plt.figure(figsize=(100, 5), facecolor=(1, 1, 1))
for ind, save in enumerate(saves):
    legend += ["stage" + str(ind + 1)]
    plt.plot(np.arange(start=len(acc_all) - 1, stop=len(acc_all + save)), np.array([acc_all[-1]] + save))
    acc_all += save
plt.legend(legend)
# plt.show()
plt.savefig("parsed_log.png")

acc_all = [saves[1][0]]
fig = plt.figure(figsize=(15, 5), facecolor=(1, 1, 1))
ax = fig.add_subplot(1, 1, 1)
c_, ls_, al_, l_ = 'y', '-', 0.8, 'cmd1'
for ind, save in enumerate(saves[1:]):
    legend += ["stage" + str(ind + 1)]
    ax.plot(np.arange(start=len(acc_all) - 1, stop=len(acc_all + save)), np.array([acc_all[-1]] + save),
            color=c_, linestyle=ls_, alpha=al_, label=l_)
    acc_all += save
    # if save[0]<0.806 :
    #     print ind,len(acc_all)
    if ind in [14, 29, 44]:
        acc_all = [saves[1][0]]
        index = [14, 29, 44].index(ind)
        c_ = ['b', 'g', 'r'][index]
        ls_ = ['--', '-.', ':'][index]
        al_ = [0.7, 0.5, 0.4][index]
        l_ = ["cmd2", "cmd3", "cmd4"][index]
# plt.legend(legend)
# plt.grid(b=True, which='major', color='k', linestyle='-')
# plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
plt.minorticks_on()
handles, labels = ax.get_legend_handles_labels()
display = (0, 15, 30, 45)

# Create legend from custom artist/label lists
ax.legend([handle for i, handle in enumerate(handles) if i in display],
          [label for i, label in enumerate(labels) if i in display])
plt.savefig("large.png")

# ------
# ------
acc_all = saves[0] + [saves[1][0]]
fig = plt.figure(figsize=(15, 5), facecolor=(1, 1, 1))
ax = fig.add_subplot(1, 1, 1)
c_, ls_, al_, l_ = 'y', '-', 0.8, 'cmd1'
ax.plot(np.arange(start=0, stop=len(acc_all)),
        np.array([acc_all]).reshape((21,)), color=c_, linestyle=ls_, alpha=al_, label=l_)
for ind, save in enumerate(saves[1:]):
    ax.plot(np.arange(start=len(acc_all) - 1, stop=len(acc_all + save)), np.array([acc_all[-1]] + save),
            color=c_, linestyle=ls_, alpha=al_, label=l_)
    acc_all += save
    # if save[0]<0.806 :
    #     print ind,len(acc_all)
    if ind in [14, 29, 44]:
        acc_all = saves[0] + [saves[1][0]]

        index = [14, 29, 44].index(ind)
        c_ = ['b', 'g', 'r'][index]
        ls_ = ['--', '-.', ':'][index]
        al_ = [0.7, 0.5, 0.4][index]
        l_ = ["cmd2", "cmd3", "cmd4"][index]
        ax.plot(np.arange(start=0, stop=len(acc_all)),
                np.array([acc_all]).reshape((21,)),
                color=c_, linestyle=ls_, alpha=al_, label=l_)
# plt.legend(legend)
# plt.grid(b=True, which='major', color='k', linestyle='-')
# plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
plt.minorticks_on()
handles, labels = ax.get_legend_handles_labels()
display = (0, 15 + 10, 30 + 10, 45 + 10)

# Create legend from custom artist/label lists
ax.legend([handle for i, handle in enumerate(handles) if i in display],
          [label for i, label in enumerate(labels) if i in display])
# plt.show()
plt.savefig("all.png")

# ---
# ---


acc_all = saves[0] + [saves[1][0]]
fig = plt.figure(figsize=(15, 5), facecolor=(1, 1, 1))
ax = fig.add_subplot(1, 1, 1)
c_, ls_, al_, l_ = 'y', '-', 0.8, 'cmd1'
ax.plot(np.arange(start=0, stop=len(acc_all)),
        np.array([acc_all]).reshape((21,)), color=c_, linestyle=ls_, alpha=al_, label=l_)
for ind, save in enumerate(saves[1:]):
    ax.plot(np.arange(start=len(acc_all) - 1, stop=len(acc_all + save)), np.array([acc_all[-1]] + save),
            color=c_, linestyle=ls_, alpha=al_, label=l_)
    acc_all += save
    # if save[0]<0.806 :
    #     print ind,len(acc_all)
    if ind in [14, 29, 44]:
        acc_all = saves[0] + [saves[1][0]]

        index = [14, 29, 44].index(ind)
        c_ = ['b', 'g', 'r'][index]
        ls_ = ['--', '-.', ':'][index]
        al_ = [0.7, 0.5, 0.4][index]
        l_ = ["cmd2", "cmd3", "cmd4"][index]
        ax.plot(np.arange(start=0, stop=len(acc_all)),
                np.array([acc_all]).reshape((21,)),
                color=c_, linestyle=ls_, alpha=al_, label=l_)
# plt.legend(legend)
# plt.grid(b=True, which='major', color='k', linestyle='-')
# plt.grid(b=True, which='minor', color='r', linestyle='-', alpha=0.2)
plt.minorticks_on()
handles, labels = ax.get_legend_handles_labels()
display = (0, 15 + 10, 30 + 10, 45 + 10)

# Create legend from custom artist/label lists
ax.legend([handle for i, handle in enumerate(handles) if i in display],
          [label for i, label in enumerate(labels) if i in display])
# plt.show()
plt.xlim([18, 175])
plt.ylim([0.77, 0.875])
plt.savefig("large.png")
