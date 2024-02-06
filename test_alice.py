from ray import tune
from imports.params import p
from tune_search import ray_train
import matplotlib.pyplot as plt
import numpy as np


# import torch
# nnmodel = torch.load(f"{p['model_path']}search by hand/obs_model_2chanbyhand29.pt", map_location=torch.device("cpu"))
# print(nnmodel)
# print(sum(params.numel() for params in nnmodel.parameters()))
# dassadads

p["search_alg"] = "Optuna"
restored_tuner = tune.Tuner.restore(p["ray_log_path"]+"/"+p["search_alg"], trainable=ray_train)
result_grid = restored_tuner.get_results()
print(result_grid.get_dataframe())
for result in result_grid:
    try:
        val_loss = result.metrics_dataframe["val loss"]
        plt.plot(range(len(val_loss)), val_loss, color="blue", label=p["search_alg"])
    except:
        pass


p["search_alg"] = "BOHB"
restored_tuner = tune.Tuner.restore(p["ray_log_path"]+"/"+p["search_alg"], trainable=ray_train)
result_grid = restored_tuner.get_results()
print(result_grid.get_dataframe())
for result in result_grid:
    try:
        val_loss = result.metrics_dataframe["val loss"]
        plt.plot(range(len(val_loss)), val_loss, color="red", label=p["search_alg"])
    except:
        pass



plt.xlabel("epoch")
plt.ylabel("val loss")
plt.yscale("log")
plt.ylim(5e-3, 2e-1)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.savefig("/home/s2041340/Thesis/plots/Search results", dpi=300)