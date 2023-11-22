from imports.data import Data
from imports.params import p



sw_path = "flamingo_0077/flamingo_0077.hdf5"
save_loc = p["base_data_path"] + ""
data = Data(p, soap_path=p['soapfile'], sw_path=sw_path)
data.create_obs_data(save_loc=save_loc, nr_samples=2000)