from os import listdir
import json
from imports.params import p
import matplotlib.pyplot as plt
import numpy as np
from imports.utility import *



l = [["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_JETS_published"],
["HYDRO_FIDUCIAL", "HYDRO_WEAK_AGN", "HYDRO_STRONG_AGN", "HYDRO_STRONGER_AGN", "HYDRO_STRONGEST_AGN" ],
["HYDRO_FIDUCIAL", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA"]]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
names = ["Jets", "fgas", "sn"]

for i, plots in enumerate(l):
    plt.figure()
    for j, model in enumerate(plots):
        p_temp = p.copy()
        p_temp["model"] = model
        filename = p_to_filename(p_temp) + ".json"
        filepath = open(p_temp['model_path'] + filename, 'r')
        d = json.load(filepath)
        plt.plot(range(d["nr_epochs"]), d["vallosses"], label=model, c=colors[j])
        plt.plot(range(d["nr_epochs"]), d["trainlosses"], c=colors[j], ls="dashed")
    plt.title("val losses")
    plt.xlabel("Epochs")
    plt.ylabel("RMSE loss")
    plt.yscale("log")
    plt.ylim(1e-3, 3e-2)
    plt.plot([0], [0], c="black", label="train losses", ls="dashed")
    plt.legend()
    plt.savefig(f"/home/s2041340/Thesis/plots/losses_{names[i]}", dpi=200)
    plt.close()