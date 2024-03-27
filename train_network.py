# from imports.networks import *
# from imports.data import *
from imports.params import p
from imports.utility import *
from imports.architectures import get_architecture
from imports.networks import *
from ray import train
import time
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Which simulation model to use")
parser.add_argument("-c", "--channel", help="Which channel: '2chan', 'low' or 'high'")
parser.add_argument("-t", "--type", help="Which type of run: 'all', 'all_but', 'single'(default)")
args = parser.parse_args()
if args.model:
    p["model"] = args.model
if args.channel:
    p["channel"] = args.channel
if args.type:
    p["simtype"] = args.type

if p["simtype"] != "single":
    name_addon = "_"+p["simtype"]
else:
    name_addon = ""

time_start = time.time()


# from ray import tune
# from tune_search import ray_train
# p["search_alg"] = "Optuna"
# restored_tuner = tune.Tuner.restore(p["ray_log_path"]+"/"+p_to_filename(p), trainable=ray_train)
# best_result = restored_tuner.get_results().get_best_result(metric="val loss", mode="min")
# params = ["lr", "L2", "batch_size", "convs_per_layer", "conv_layers", "use_batch_norm", "leaky_slope", "base_filters", "bn_momentum"]
# for param in params:
#     p[param] = best_result.config[param]

p["nr_epochs"] = 1
p["batch_size"] = 512

p["lr"] = 3.33414e-05
p["L2"] = 0.00133171
# p["batch_size"] = 512
p["convs_per_layer"] = 1
p["conv_layers"] = 1
p["use_batch_norm"] = False
p["leaky_relu"] = 0.03
p["kernel_size"] = 3
p["dropout"] = 0.1
p["base_filters"] = 64


# p["batch_size"] = 64

# p["soapfile"] = "halo_properties_0078.hdf5"
# p["snapshot"] = "flamingo_0078/flamingo_0078.hdf5"
# p["snapshot_folder"] = "snapshots_reduced"
# p["simsize"] = "L2800N5040"



p["architecture"] = get_architecture(p)
print("architecture: ")
for layer in p["architecture"]:
    print(layer)
print("p: ")
for key in p:
    print(key+":", p[key])

torch.cuda.memory._record_memory_history()


model, p = train_network(p, report=False)
print(p, flush=True)

# modelname = p_to_filename(p)
# torch.save(model, p["model_path"]+modelname+".pt")

# with open(p['model_path'] + modelname + ".json", 'w') as filepath:
#     json.dump(p, filepath, indent=4)
# print("File name:", modelname)
# print("Time spent: {}s, or {}m".format(time.time() - time_start, (time.time() - time_start)/60))


torch.cuda.memory._dump_snapshot("my_snapshot_train5.pickle")
