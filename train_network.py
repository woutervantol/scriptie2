from imports.networks import *
from imports.data import *
from imports.params import p
from imports.utility import *
from imports.architectures import get_architecture
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Which simulation model to use")
parser.add_argument("-c", "--channel", help="Which channel: '2chan', 'low' or 'high'")
args = parser.parse_args()
if args.model:
    p["model"] = args.model
if args.channel:
    p["channel"] = args.channel

time_start = time.time()



from ray import tune
from tune_search import ray_train
p["search_alg"] = "Optuna"
restored_tuner = tune.Tuner.restore(p["ray_log_path"]+"/"+p_to_filename(p)+"_all", trainable=ray_train)
print(restored_tuner)
best_result = restored_tuner.get_results().get_best_result(metric="val loss", mode="min", scope="all")
print(best_result)
params = ["lr", "L2", "batch_size", "convs_per_layer", "conv_layers", "use_batch_norm", "leaky_slope", "base_filters", "bn_momentum"]
for param in params:
    p[param] = best_result.config[param]

p["lrfactor"] = 0.7
p["lrpatience"] = 10
p["nr_epochs"] = 300

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


sw_path = "flamingo_0077/flamingo_0077.hdf5"
data = Data(p)
filename = []
# for i in range(1, 6):
for model in ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_JETS_published", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
    p_temp = p.copy()
    p_temp["model"] = model
    filename.append(p_to_filename(p_temp))
# filename = p_to_filename(p)
data.make_nn_dataset(filename=filename, target="TotalMass")

model = Model(p)
model.set_convolutional_model()
model.set_optimizer()

model.train(data, verbose=2)

p["trainlosses"] = model.losses
p["vallosses"] = model.val_losses
p["lrs"] = model.lrs
modelname = p_to_filename(p) + "_all"
torch.save(model.model, p['model_path'] + modelname + ".pt")

import json
with open(p['model_path'] + modelname + ".json", 'w') as filepath:
    json.dump(p, filepath, indent=4)

print("File name:", modelname)
print("Time spent: {}s, or {}m".format(time.time() - time_start, (time.time() - time_start)/60))




# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("-m", "--model", help="Which simulation model to use")
# args = parser.parse_args()
# if args.model:
#     p["model"] = args.model


#####old code, parameter search
# count = 0
# layers = []
# for channel in ["2chan", "low", "high"]:
#     for conv_channels in [16, 64]:
#         for leaky_slope in [0.0, 0.01]:
#             for dropout in [0.0, 0.2]:
#                 for conv_layers in [1, 4]:
#                     for use_pooling in [False, True]:
#                         for use_batch_norm in [False, True]:
#                             time_start = time.time()
#                             p["channel"] = channel
#                             p["conv_channels"] = conv_channels
#                             p["leaky_slope"] = leaky_slope
#                             p["dropout"] = dropout
#                             p["conv_layers"] = conv_layers
#                             p["use_pooling"] = use_pooling
#                             p["use_batch_norm"] = use_batch_norm

#                             p["architecture"] = get_architecture(p)
#                             model = Model(p)
#                             model.set_convolutional_model()
#                             model.set_optimizer()
#                             model.train(data, verbose=2)

#                             p["time_to_train"] = time.time() - time_start
#                             p["trainlosses"] = model.losses
#                             p["vallosses"] = model.val_losses
#                             p["lrs"] = model.lrs
#                             modelname = f"obs_model_" + p['channel'] + f"_{count}"
#                             torch.save(model.model, p['model_path'] + modelname + ".pt")

#                             import json
#                             with open(p['model_path'] + modelname + ".json", 'w') as filepath:
#                                 json.dump(p, filepath, indent=4)
                            

#                             print(f"Model {count} done in {time.time() - time_start} seconds")
#                             print(f"Lowest val loss: {np.min(model.val_losses)}, lowest train loss: {np.min(model.losses)}")
#                             count += 1