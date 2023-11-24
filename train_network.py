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



sw_path = "flamingo_0077/flamingo_0077.hdf5"
data = Data(p, sw_path=sw_path)
filepath = f"{p['base_data_path']}/obs_data_{p_to_filename(p)}_M1e13_rad2Mpc"
data.make_obs_dataset(filepath=filepath, channel=p["channel"], target="TotalMass")


p["architecture"] = get_architecture(use_model, channel=p["channel"])
model = Model(p, lr=p["lr"], batch_size=p["batch_size"])
model.set_convolutional_model(p["architecture"])
model.set_optimizer()
model.train(data, nr_epochs=p["nr_epochs"])


p["trainlosses"] = model.losses
p["vallosses"] = model.val_losses
torch.save(model.model, f"{p['base_model_path']}/obs_model_{p['channel']}.pt")
import json
with open(f"{p['base_model_path']}/obs_model_{p['channel']}.json", 'w') as filepath:
    json.dump(p, filepath, indent=4)