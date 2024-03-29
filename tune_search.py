import numpy as np
import torch
import time
from imports.params import p
from imports.utility import *
from ray import train
from imports.networks import CustomCNN, train_network
from imports.architectures import get_architecture
from ray import tune
import ray
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Which simulation model to use")
parser.add_argument("-c", "--channel", help="Which channel: '2chan', 'low' or 'high'")
parser.add_argument("-t", "--type", help="Which type of run: 'all', 'all_but', 'single'(default)")
parser.add_argument("-b", "--budget", help="Time budget for single tune experiment, in hours.")
args = parser.parse_args()
if args.model:
    p["model"] = args.model
if args.channel:
    p["channel"] = args.channel
if args.type:
    p["simtype"] = args.type
if args.budget:
    p["time_budget"] = 60*60*float(args.budget)

if p["simtype"] != "single":
    name_addon = "_"+p["simtype"]
else:
    name_addon = ""

p["search_alg"] = "Optuna"
# p["time_budget"] = 60*60*3.5
p["nr_epochs"] = 300

# p["soapfile"] = "halo_properties_0078.hdf5"
# p["snapshot"] = "flamingo_0078/flamingo_0078.hdf5"
# p["snapshot_folder"] = "snapshots_reduced"
# p["simsize"] = "L2800N5040"


print("p: ")
for key in p:
    print(key+":", p[key])

config = {
    "lr": tune.loguniform(3e-6, 1e-2),
    "L2": tune.loguniform(0.001, 0.01),
    "batch_size": tune.choice([64, 128, 256, 512]),
    "convs_per_layer": tune.choice([1, 2, 3]),
    "conv_layers":tune.choice([1, 2, 3]),
    "use_batch_norm": tune.choice([True, False]),
    "leaky_slope": tune.choice([0.0, 0.01, 0.03, 0.1]),
    "base_filters": tune.choice([16, 32, 64, 128]),
    "bn_momentum": tune.loguniform(0.001, 0.2),
    "dropout": tune.choice([0.0, 0.1, 0.4]),
    "kernel_size": tune.choice([3, 5])
}

points_to_evaluate=[{"lr":0.0001, 
                     "L2":0.003, 
                     "batch_size":64, 
                     "convs_per_layer":1,
                     "conv_layers":1,
                     "use_batch_norm":False,
                     "leaky_slope":0.0,
                     "base_filters": 64,
                     "bn_momentum": 0.1,
                     "dropout": 0.0,
                     "kernel_size": 3}]


# if p["search_alg"] == "BOHB":
#     from ray.tune.search.bohb import TuneBOHB
#     search_alg = TuneBOHB(metric="val loss", mode="min", points_to_evaluate=points_to_evaluate)
#     scheduler = tune.schedulers.HyperBandForBOHB(time_attr="time_total_s", metric="val loss", mode="min")

if p["search_alg"] == "Optuna":
    from ray.tune.search.optuna import OptunaSearch
    ### OptunaSearch default is gewoon TPE, met n_startup_trials=10, n_ei_candidates=24
    search_alg = OptunaSearch(metric="val loss", mode="min", points_to_evaluate=points_to_evaluate)
    # scheduler = tune.schedulers.MedianStoppingRule(time_attr="time_total_s", metric="val loss", mode="min", grace_period=10, hard_stop=True, min_samples_required=3)
    scheduler = tune.schedulers.ASHAScheduler(time_attr="time_total_s", metric="val loss", mode="min", max_t=1000, grace_period=100)

# if p["search_alg"] == "HyperOpt":
#     from ray.tune.search.hyperopt import HyperOpt
#     search_alg = HyperOpt(metric="val loss", mode="min", points_to_evaluate=points_to_evaluate)
#     scheduler = tune.schedulers.ASHAScheduler(time_attr="training_iteration", metric="val loss", mode="min", max_t=100, grace_period=3)



def ray_train(config, p):
    p["lr"] = config["lr"]
    p["L2"] = config["L2"]
    p["batch_size"] = config["batch_size"]
    p["convs_per_layer"] = config["convs_per_layer"]
    p["conv_layers"] = config["conv_layers"]
    p["use_batch_norm"] = config["use_batch_norm"]
    p["leaky_slope"] = config["leaky_slope"]
    p["base_filters"] = config["base_filters"]
    p["bn_momentum"] = config["bn_momentum"]
    p["dropout"] = config["dropout"]
    p["kernel_size"] = config["kernel_size"]
    operation_size = p["resolution"]*p["resolution"]*p["batch_size"]*p["base_filters"]*4 #32-bit numbers=4bytes
    layer_size = operation_size * (p["convs_per_layer"]+1) * 3 #assumes conv+leakyrelu+batchnorm per conv
    net_size = layer_size * (p["conv_layers"]+1)/2 #semi accurate for up to 3 layers
    if net_size > 9e9:
        train.report({"val loss": np.inf, "loss":np.inf})
        return

    train_network(p, verbose=0, report=True)

def run():
    try:
        ray.init("auto")
    except:
        pass
    tuner = tune.Tuner(tune.with_resources(tune.with_parameters(ray_train, p=p), resources={"gpu":1}), 
                        param_space=config, 
                        tune_config=tune.TuneConfig(scheduler=scheduler, 
                                                    search_alg=search_alg, 
                                                    num_samples=1000, 
                                                    time_budget_s=p["time_budget"]),
                        run_config=train.RunConfig(storage_path=p["ray_log_path"], 
                                                   name=p_to_filename(p)+name_addon, 
                                                   progress_reporter=tune.CLIReporter(max_progress_rows=3),)
                                                #    checkpoint_config=train.CheckpointConfig(1, "val loss", "min"))
                       )


    result_grid = tuner.fit()
    best_result = result_grid.get_best_result(metric="val loss", mode="min", scope="all")

    print(best_result)
    print(best_result.metrics, flush=True)


if __name__=="__main__":
    run()


# def train(p):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     p["architecture"] = get_architecture(p)
#     model = CustomCNN(p["architecture"]).to(device)
#     trainx, trainy, valx, valy = ray_get_data(p)
#     optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["L2"])
#     gamma = (1./100.)**(1./300.)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
#     # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=p["lrfactor"], patience=p["lrpatience"])
#     for epoch in range(p["nr_epochs"]):
#         epoch_start = time.time()
#         nr_batches = int(len(trainy)/config["batch_size"])
#         trainx, trainy = ray_shuffle(trainx, trainy)
#         train_losses = 0
#         for batch in range(nr_batches):
#             batch_start = batch*config["batch_size"]
#             batch_stop = (batch+1)*config["batch_size"]
#             y_pred = model(torch.Tensor(trainx[batch_start:batch_stop]).to(device)).squeeze(1)
#             target = torch.Tensor(trainy[batch_start:batch_stop]).to(device)
#             loss = ray_MSE(y_pred, target)
#             with torch.no_grad():
#                 train_losses += np.float64(loss.cpu())
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()
        
        
#         with torch.no_grad():
#             indices = np.arange(len(valx))
#             max_val_samples = 2000
#             selection = np.random.choice(indices, min(len(valx), max_val_samples), replace=False)
#             val_pred = model(torch.Tensor(valx[selection]).to(device)).squeeze(1)
#             val_true = torch.Tensor(valy[selection]).to(device)
#             val_loss = np.float64(ray_MSE(val_pred, val_true).cpu())
#             scheduler.step(val_loss)
#             train.report({"val loss": val_loss, "loss":np.float64(loss.cpu()), "lr":optimizer.param_groups[-1]["lr"]})
#     print(f"Reached epoch {epoch} of {p['nr_epochs']}", flush=True)
#         # if time.time() - epoch_start > 100:
#         #     return

# def ray_shuffle(datax, datay):
#     indices = np.arange(len(datay))
#     np.random.shuffle(indices)
#     return datax[indices], datay[indices]

# def ray_MSE(pred, true):
#     return (pred - true).square().mean()


# def ray_get_data(p):
#     filename =  p_to_filename(p)
#     data_x = np.empty((0, 2, p["resolution"], p["resolution"]))
#     data_y = np.empty((0))
#     if p["simtype"] != "single":
#         for model in ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_JETS_published", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
#             if p["simtype"] == "all_but" and p["model"] == model:
#                 continue
#             p_temp = p.copy()
#             p_temp["model"] = model
#             filename = p_to_filename(p_temp)
#             data_x = np.append(data_x, np.load(p["data_path"] + filename + ".npy"), axis=0)
#             data_y = np.append(data_y, np.load(p["data_path"] + filename + "_masses.npy"), axis=0)
#     else:
#         data_x = np.load(p["data_path"] + filename + ".npy")
#         data_y = np.load(p["data_path"] + filename + "_masses.npy")

#     data_x = np.log10(data_x)
#     data_y = np.log10(data_y)
#     std_x = np.std(data_x, axis=(0, 2, 3))
#     std_y = np.std(data_y)
#     mean_x = np.mean(data_x, axis=(0, 2, 3))
#     mean_y = np.mean(data_y)

#     #scale and shift data for better nn training
#     data_x = (data_x - mean_x[np.newaxis, :, np.newaxis, np.newaxis]) / std_x[np.newaxis, :, np.newaxis, np.newaxis]
#     data_y = (data_y - mean_y) / std_y
    
#     #Select the correct image for single channel runs
#     if p["channel"]=="low": 
#         data_x = data_x[:,:1,:,:]
#     elif p["channel"]=="high": 
#         data_x = data_x[:,1:,:,:]
#     else:
#         pass

#     shuffled_indices = np.arange(len(data_y))
#     np.random.shuffle(shuffled_indices)
#     data_x = data_x[shuffled_indices]
#     data_y = data_y[shuffled_indices]

#     test_split = int(len(data_y)*p["test_size"])
#     val_split = test_split + int(len(data_y)*p["val_size"])
#     valx = data_x[test_split:val_split]
#     valy = data_y[test_split:val_split]
#     trainx = data_x[val_split:]
#     trainy = data_y[val_split:]
    
#     return trainx, trainy, valx, valy

