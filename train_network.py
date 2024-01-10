from imports.networks import *
from imports.data import *
from imports.params import p
from imports.utility import *
from imports.architectures import get_architecture
print("imports done")

p["channel"] = "2chan"
p["lr"] = 0.0001
p["lrfactor"] = 0.5
p["lrpatience"] = 10
p["L2"] = 0.0 #0.01
p["batch_size"] = 64
p["nr_epochs"] = 200
# p["conv_channels"] = 64
# p["conv_layers"] = 4
p["leaky_slope"] = 0.0
# p["dropout"] = 0.0
# p["use_pooling"]=False
# p["use_batch_norm"]=False
p["architecture"] = get_architecture(p)

print("architecture loaded")

p["channel"] = "2chan"
p["architecture"] = get_architecture(p)

sw_path = "flamingo_0077/flamingo_0077.hdf5"
data = Data(p, sw_path="")
filename = p_to_filename(p) + "big2"
data.make_nn_dataset(filename=filename, target="TotalMass")
print("dataset made")

model = Model(p)
model.set_convolutional_model()
model.set_optimizer()
model.train(data, verbose=2)

p["trainlosses"] = model.losses
p["vallosses"] = model.val_losses
p["lrs"] = model.lrs
modelname = f"obs_model_" + p['channel'] + "byhand"
torch.save(model.model, p['model_path'] + modelname + ".pt")

import json
with open(p['model_path'] + modelname + ".json", 'w') as filepath:
    json.dump(p, filepath, indent=4)





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