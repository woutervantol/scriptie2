from imports.networks import *
from imports.data import *
from imports.params import p
from imports.utility import *
from imports.architectures import get_architecture

p["channel"] = "2chan"
p["lr"] = 0.00001
p["batch_size"] = 16
p["nr_epochs"] = 50
use_model = "base_conv_network"
p["architecture"] = get_architecture(use_model, channel=p["channel"])




sw_path = "flamingo_0077/flamingo_0077.hdf5"
data = Data(p, sw_path=sw_path)
filename = p_to_filename(p) + "_M1e13_rad2Mpc"
data.make_obs_dataset(filename=filename, channel=p["channel"], target="TotalMass")


model = Model(p)
model.set_convolutional_model()
model.set_optimizer()
model.train(data)


p["trainlosses"] = model.losses
p["vallosses"] = model.val_losses
modelname = f"obs_model_" + p['channel']
torch.save(model.model, p['model_path'] + modelname + ".pt")
import json
with open(p['model_path'] + modelname + ".json", 'w') as filepath:
    json.dump(p, filepath, indent=4)