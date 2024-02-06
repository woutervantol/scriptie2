from imports.networks import *
from imports.data import *
from imports.params import p
from imports.utility import *
from imports.architectures import get_architecture


time_start = time.time()
p["channel"] = "2chan"
p["lrfactor"] = 0.7
p["lrpatience"] = 10
p["nr_epochs"] = 300

# p["lr"] = 0.0008
# p["L2"] = 0.003
# p["batch_size"] = 512
# p["convs_per_layer"] = 2
# p["conv_layers"] = 1
# p["use_batch_norm"] = False
# p["leaky_slope"] = 0.0
# p["base_filters"] = 64
# p["bn_momentum"] = 0.01


from ray import tune
from tune_search import ray_train
p["search_alg"] = "Optuna"
restored_tuner = tune.Tuner.restore(p["ray_log_path"]+"/"+p["search_alg"], trainable=ray_train)
best_result = restored_tuner.get_results().get_best_result(metric="val loss", mode="min")
params = ["lr", "L2", "batch_size", "convs_per_layer", "conv_layers", "use_batch_norm", "leaky_slope", "base_filters", "bn_momentum"]
for param in params:
    p[param] = best_result.config[param]


p["architecture"] = get_architecture(p)
print("architecture: ")
for layer in p["architecture"]:
    print(layer)
print("p: ")
for key in p:
    print(key+":", p[key])


sw_path = "flamingo_0077/flamingo_0077.hdf5"
data = Data(p)
filename = p_to_filename(p) + "big2"
data.make_nn_dataset(filename=filename, target="TotalMass")

model = Model(p)
model.set_convolutional_model()
model.set_optimizer()

model.train(data, verbose=2)

p["trainlosses"] = model.losses
p["vallosses"] = model.val_losses
p["lrs"] = model.lrs
modelname = f"obs_model_" + p['channel'] + "best_bohb"
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
#     p["cosmology"] = args.model


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