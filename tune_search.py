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

### define arguments for the slurm file to vary
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Which simulation model to use")
parser.add_argument("-c", "--channel", help="Which channel: '2chan', 'low' or 'high'")
parser.add_argument("-t", "--type", help="Which type of run: 'all', 'all_but', 'single'(default)")
parser.add_argument("-b", "--budget", help="Time budget for single tune experiment, in hours.")
parser.add_argument("-n", "--noise", help="Use noise or not, Bool value.")
args = parser.parse_args()
if args.model:
    p["model"] = args.model
if args.channel:
    p["channel"] = args.channel
if args.type:
    p["simtype"] = args.type
if args.budget:
    p["time_budget"] = 60*60*float(args.budget)

### change file naming based on arguments
if p["simtype"] != "single":
    name_addon = "_"+p["simtype"]
else:
    name_addon = ""
if p["channel"] != "2chan":
    name_addon += "_"+p["channel"]
if bool(args.noise):
    p["noisy"] = True
    name_addon += "_"+"noisy"

### set parameters
p["search_alg"] = "Optuna"
p["nr_epochs"] = 300 # max nr of epochs if it is reached within the early stopping time or within the trial time budget
p["redshift"] = 0.15

### large volume parameters
# p["soapfile"] = "halo_properties_0078.hdf5"
# p["snapshot"] = "flamingo_0078/flamingo_0078.hdf5"
# p["snapshot_folder"] = "snapshots_reduced"
# p["simsize"] = "L2800N5040"

### print parameters for identifying logs
print("p: ")
for key in p:
    print(key+":", p[key], flush=True)

### define hyperparameter ranges
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

### define single trial to try first for benchmarking. Found emperically
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


if p["search_alg"] == "BOHB":
    from ray.tune.search.bohb import TuneBOHB
    search_alg = TuneBOHB(metric="val loss", mode="min", points_to_evaluate=points_to_evaluate)
    scheduler = tune.schedulers.HyperBandForBOHB(time_attr="time_total_s", metric="val loss", mode="min")
elif p["search_alg"] == "Optuna": #default
    from ray.tune.search.optuna import OptunaSearch
    search_alg = OptunaSearch(metric="val loss", mode="min", points_to_evaluate=points_to_evaluate)
    scheduler = tune.schedulers.ASHAScheduler(time_attr="time_total_s", metric="val loss", mode="min", max_t=1000, grace_period=100)
elif p["search_alg"] == "HyperOpt":
    from ray.tune.search.hyperopt import HyperOpt
    search_alg = HyperOpt(metric="val loss", mode="min", points_to_evaluate=points_to_evaluate)
    scheduler = tune.schedulers.ASHAScheduler(time_attr="time_total_s", metric="val loss", mode="min", max_t=1000, grace_period=100)



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
    
    ### approximate memory usage based on parameters to prevent memory overflow
    operation_size = p["resolution"]*p["resolution"]*p["batch_size"]*p["base_filters"]*4 #32-bit numbers=4bytes
    layer_size = operation_size * (p["convs_per_layer"]+1) * 3 #assumes conv+leakyrelu+batchnorm per conv
    net_size = layer_size * (p["conv_layers"]+1)/2 #semi accurate for up to 3 layers
    if net_size > 9e9:
        train.report({"val loss": np.inf, "loss":np.inf})
        return
    train_network(p, verbose=0, report=True)


def run():
    print(p["model"], flush=True)

    ### create head and initialise ray
    ray.init(num_cpus=4, num_gpus=1) #cpu and gpu amount must be explicitly stated to prevent error

    ### initialise and tune worker
    tuner = tune.Tuner(tune.with_resources(tune.with_parameters(ray_train, p=p), resources={"gpu":1, "cpu":4}), 
                        param_space=config, 
                        tune_config=tune.TuneConfig(scheduler=scheduler, 
                                                    search_alg=search_alg, 
                                                    num_samples=1000, 
                                                    time_budget_s=p["time_budget"]),
                        run_config=train.RunConfig(storage_path=p["ray_log_path"], 
                                                   name=p_to_filename(p)+name_addon, 
                                                   progress_reporter=tune.CLIReporter(max_progress_rows=3))
                       )

    ### run tune worker. Automatically saves best state of the model
    result_grid = tuner.fit()
    best_result = result_grid.get_best_result(metric="val loss", mode="min", scope="all")

    print(best_result)
    print(best_result.metrics, flush=True)


if __name__=="__main__":
    run()