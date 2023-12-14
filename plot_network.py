from os import listdir
import json
from imports.params import p
import matplotlib.pyplot as plt
import numpy as np

flist = listdir(p["model_path"])

files = []
for file in flist:
    if "obs_model_2chan_" in file and ".json" in file:
        files.append(file)


best = 1000.0
fig, ax = plt.subplots()
for modelname in files:
    filepath = open(p['model_path'] + modelname, 'r')
    d = json.load(filepath)
    if np.min(d["vallosses"]) < best:
        best = np.min(d["vallosses"])
        print(modelname)
    # c = "blue" if d["use_batch_norm"] else "red"
    # l = "Batch Norm" if d["use_batch_norm"] else "No Batch Norm"
#     ax.plot(range(d["nr_epochs"]), d["vallosses"], c=c, label=l)
# ax.set_yscale("log")
# ax.set_xlabel("Epoch")
# ax.set_ylabel("MSE loss")
# handles, labels = plt.gca().get_legend_handles_labels()
# by_label = dict(zip(labels, handles))
# plt.legend(by_label.values(), by_label.keys())
# plt.title("Validation loss curves for all parameter combinations")
# plt.savefig(p["data_path"] + "plots/loss_use_batch_norm", dpi=300)