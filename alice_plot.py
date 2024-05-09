from imports.params import p
import matplotlib.pyplot as plt
import numpy as np
from imports.utility import *
import torch
from imports.data import Data

def run():
    from imports.params import p

    p["simtype"] = "single"
    for model in ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_JETS_published", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
        p["model"] = model
        data = Data(p)
        plot_single_rmse(p, data)
    p["simtype"] = "all"
    
# getraind op ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_JETS_published", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]
# addon "all", "all_but", "", "extremes"
# test_data = ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_JETS_published", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]
# "rmse" of "scatter"



# def plot_errors(masses, predictions, label, color):
#     plt.scatter(np.log10(masses), np.log10(predictions) - np.log10(masses), label=label, color=color, s=2)
#     plt.ylabel("$\log_{10}(M_{pred}) - \log_{10}(M_{true})$")
#     plt.xlabel("$\log_{10} M_{true}$")
#     # plt.title(p["model"])
#     plt.legend()
#     # plt.savefig("/home/s2041340/Thesis/plots/error_"+p_to_filename(p)+"_scalingrelation", dpi=200)
#     # plt.close()


# def plot_rmse(masses, predictions, label, color):
#     from scipy.stats import bootstrap
#     nr_bins = 10
#     x = np.array([])
#     y = np.array([])
#     perc16 = np.array([])
#     perc84 = np.array([])
#     bin_edges = np.logspace(np.log10(np.min(masses)), np.log10(np.max(masses)), nr_bins+1)
#     for b in range(nr_bins):
#         indices = np.logical_and(masses > bin_edges[b], masses < bin_edges[b+1])
#         error = np.log10(predictions[indices]) - np.log10(masses[indices])
#         x = np.append(x, np.log10(np.mean([bin_edges[b], bin_edges[b+1]])))
#         y = np.append(y, rmse(error))
#         b = bootstrap((error,), rmse, confidence_level=0.84)
#         perc16 = np.append(perc16, b.confidence_interval.low)
#         perc84 = np.append(perc84, b.confidence_interval.high)
#     plt.plot(x, y, label=label, color=color)
#     plt.fill_between(x, perc16, perc84, color=color, alpha=0.2)
#     plt.xlabel("M$_{500c}$ $(M_\odot)$")
#     plt.ylabel("Root Mean Squared Error")
#     # plt.title(p["model"])
#     plt.legend()
#     plt.ylim(0, 0.15)
#     # plt.savefig("/home/s2041340/Thesis/plots/RMSE_"+p_to_filename(p), dpi=200)
#     # plt.close()


def rmse(d):
    return np.sqrt(np.mean((d)**2))

def plot_single_rmse(p, data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.make_nn_dataset(p_to_filename(p))
    nnmodel = torch.load(f"{p['model_path']}/{p_to_filename(p)}.pt").to(device)
    with torch.no_grad():
        nn_predictions = np.array(10**(nnmodel(torch.Tensor(data.testx).to(device)).squeeze(1) * data.std_y + data.mean_y).cpu())

    data.load_testset(p_to_filename(p))
    predictions_low, predictions_high = predict_linear(p, data)

    names = ["NN", "Spline low", "Spline high"]
    colors = ["green", "m", "y"]

    for i, predictions in enumerate([nn_predictions, predictions_low, predictions_high]):
        plt.scatter(np.log10(data.masses), np.log10(predictions) - np.log10(data.masses), label=names[i], color=colors[i], s=2)
    plt.ylabel("$\log_{10}(M_{pred}) - \log_{10}(M_{true})$")
    plt.xlabel("$\log_{10} M_{true}$")
    plt.title(p["model"])
    plt.legend()
    plt.savefig("/home/s2041340/Thesis/plots/error_"+p_to_filename(p)+"_scalingrelation", dpi=200)
    plt.close()

    from scipy.stats import bootstrap
    for i, predictions in enumerate([nn_predictions, predictions_low, predictions_high]):
        nr_bins = 10
        x = np.array([])
        y = np.array([])
        perc16 = np.array([])
        perc84 = np.array([])
        bin_edges = np.logspace(np.log10(np.min(data.masses)), np.log10(np.max(data.masses)), nr_bins+1)
        for b in range(nr_bins):
            indices = np.logical_and(data.masses > bin_edges[b], data.masses < bin_edges[b+1])
            error = np.log10(predictions[indices]) - np.log10(data.masses[indices])
            x = np.append(x, np.log10(np.mean([bin_edges[b], bin_edges[b+1]])))
            y = np.append(y, rmse(error))
            b = bootstrap((error,), rmse, confidence_level=0.84)
            perc16 = np.append(perc16, b.confidence_interval.low)
            perc84 = np.append(perc84, b.confidence_interval.high)
        plt.plot(x, y, label=names[i], color=colors[i])
        plt.fill_between(x, perc16, perc84, color=colors[i], alpha=0.2)
    plt.xlabel("M$_{500c}$ $(M_\odot)$")
    plt.ylabel("Root Mean Squared Error")
    plt.title(p["model"])
    plt.legend()
    plt.ylim(0, 0.15)
    plt.savefig("/home/s2041340/Thesis/plots/RMSE_"+p_to_filename(p), dpi=200)
    plt.close()


def predict_linear(p, data):
    spl_low = np.load(f"/home/s2041340/data1/scaling_relations/{p_to_filename(p)}_spline_fit_low.npy", allow_pickle=True)[()]
    spl_high = np.load(f"/home/s2041340/data1/scaling_relations/{p_to_filename(p)}_spline_fit_high.npy", allow_pickle=True)[()]

    pixel_size = 4 * 1000 / 64  #kpc
    radii = []
    for i, haloindex in enumerate(data.indices):
        r = data.soap_file[f"{data.selection_type}/SORadius"][haloindex] * 1000 #kpc
        radius = r / pixel_size
        radii.append(radius)
        X, Y = np.ogrid[:p['resolution'], :p['resolution']]
        dist_from_center = np.sqrt((X + 0.5-int(p['resolution']/2))**2+(Y+0.5-int(p['resolution']/2))**2)
        mask_outer = dist_from_center<=radius
        mask_inner = dist_from_center<=(radius-1)
        luminosity_circle_low = np.log10(np.sum(data.images[i, 0][mask_outer]))
        luminosity_circle_high = np.log10(np.sum(data.images[i, 1][mask_outer]))
        predictions_low = 10**spl_low(luminosity_circle_low)
        predictions_high = 10**spl_high(luminosity_circle_high)
    return predictions_low, predictions_high


def plot_losses():
    import json
    from os import listdir
    l = [["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_JETS_published"],
    ["HYDRO_FIDUCIAL", "HYDRO_WEAK_AGN", "HYDRO_STRONG_AGN", "HYDRO_STRONGER_AGN", "HYDRO_STRONGEST_AGN"],
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





if __name__=="__main__":
    run()