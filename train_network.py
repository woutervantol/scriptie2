from imports.networks import *
from imports.data import *
from imports.params import p
from imports.utility import *
from imports.architectures import get_architecture
from ray_train import ray_train
from ray import tune
from ray.train import RunConfig
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Which simulation model to use")
args = parser.parse_args()
if args.model:
    p["cosmology"] = args.model

time_start = time.time()
p["channel"] = "2chan"
p["lr"] = 0.0001
p["lrfactor"] = 0.2
p["lrpatience"] = 70
p["L2"] = 0.003 #0.01
p["batch_size"] = 64
p["nr_epochs"] = 500
# p["conv_channels"] = 64
# p["conv_layers"] = 4
p["leaky_slope"] = 0.0
# p["dropout"] = 0.3
# p["use_pooling"]=False
# p["use_batch_norm"]=False


p["architecture"] = get_architecture(p)
print("architecture loaded")


sw_path = "flamingo_0077/flamingo_0077.hdf5"
data = Data(p)
filename = p_to_filename(p) + "big2"
data.make_nn_dataset(filename=filename, target="TotalMass")
print("dataset made")

model = Model(p)
model.set_convolutional_model()
model.set_optimizer()



config = {
    "L2": tune.loguniform(0.001, 0.01),
    "lr": tune.loguniform(0.001, 0.0001),
    # "batch_size": tune.choice([64, 128, 256, 512]),
    "batch_size": tune.choice([64]),
    "leaky_slope": tune.uniform(0, 0.1),
    "barch_norm": tune.choice([True, False]),
    "nr_epochs": tune.choice([2])
}

tuner = tune.Tuner(tune.with_resources(tune.with_parameters(ray_train, net=model.model, p=p), resources={"cpu":4}), 
                    param_space=config, 
                    tune_config=tune.TuneConfig(metric="val loss", mode="min", num_samples=7, max_concurrent_trials=1),
                    run_config=RunConfig(storage_path=p["ray_log_path"], name="test", progress_reporter=tune.CLIReporter(max_progress_rows=3)))

results = tuner.fit()
print(results.get_best_result(metric="val loss", mode="min"))



# model.train(data, verbose=2)

# p["trainlosses"] = model.losses
# p["vallosses"] = model.val_losses
# p["lrs"] = model.lrs
# modelname = f"obs_model_" + p['channel'] + "lr_test"
# torch.save(model.model, p['model_path'] + modelname + ".pt")

# import json
# with open(p['model_path'] + modelname + ".json", 'w') as filepath:
#     json.dump(p, filepath, indent=4)

# print("File name:", modelname)
# print("Time spent: {}s, or {}m".format(time.time() - time_start, (time.time() - time_start)/60))



#old code, parameter search
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