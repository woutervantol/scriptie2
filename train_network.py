from imports.networks import *
from imports.data import *
from imports.params import p
from imports.utility import *

# device = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )

sw_path = "flamingo_0077/flamingo_0077.hdf5"
data = Data(p, sw_path=sw_path)
filepath = f"{p['base_data_path']}/obs_data_{p_to_filename(p)}_M1e13_rad2Mpc"
data.make_obs_dataset(filepath=filepath, channel="2chan", target="TotalMass")


model = Model(p, lr=0.00001, batch_size=16)
model.set_convolutional_model(p["base_convolutional_network"])
model.set_optimizer()

model.train(data, nr_epochs=50)


torch.save(model.model, f"{p['base_model_path']}/obs_model_2chan_M13-15_L45.pt")