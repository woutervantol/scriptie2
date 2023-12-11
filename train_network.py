from imports.networks import *
from imports.data import *
from imports.params import p
from imports.utility import *
from imports.architectures import get_architecture

p["channel"] = "2chan"
p["lr"] = 0.0001
p["lrfactor"] = 0.5
p["lrpatience"] = 10
p["L2"] = 0.01
p["nr_epochs"] = 200
p["use_pooling"]=True
p["architecture"] = get_architecture(p)

sw_path = "flamingo_0077/flamingo_0077.hdf5"
data = Data(p, sw_path=sw_path)
filename = p_to_filename(p) + "big2"
data.make_nn_dataset(filename=filename, target="TotalMass")

# for layer in p["architecture"]:
#     print(layer)
# count = 0
# layers = []
# for channel in ["2chan", "low", "high"]:
#     for kernel_size in [3, 5]:
#         for conv_channels in [16, 32, 64]:
#             for leaky_slope in [0.0, 0.01, 0.05]:
#                 for dropout in [0.0, 0.1, 0.4]:
#                     for conv_layers in [1, 2, 4]:
#                         for use_pooling in [False, True]:
#                             for use_batch_norm in [False, True]:
#                                 p["channel"] = channel
#                                 p["kernel_size"] = kernel_size
#                                 p["conv_channels"] = conv_channels
#                                 p["leaky_slope"] = leaky_slope
#                                 p["dropout"] = dropout
#                                 p["conv_layers"] = conv_layers
#                                 p["use_pooling"] = use_pooling
#                                 p["use_batch_norm"] = use_batch_norm

#                                 p["architecture"] = get_architecture(p)
#                                 model = Model(p)
#                                 model.set_convolutional_model()
#                                 model.set_optimizer()
#                                 model.train(data)
#                                 count += 1

model = Model(p)
model.set_convolutional_model()
model.set_optimizer()
model.train(data, verbose=2)

p["trainlosses"] = model.losses
p["vallosses"] = model.val_losses
p["lrs"] = model.lrs
modelname = f"obs_model_" + p['channel']
torch.save(model.model, p['model_path'] + modelname + ".pt")
import json
with open(p['model_path'] + modelname + ".json", 'w') as filepath:
    json.dump(p, filepath, indent=4)