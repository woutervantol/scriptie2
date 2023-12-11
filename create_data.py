from imports.data import Data
from imports.params import p
from imports.utility import *

p["resolution"] = 64
p["obs_radius"] = 2 #Mpc
p["nr_uniform_bins_obs_data"] = 50


sw_path = "flamingo_0077/flamingo_0077.hdf5"
filename = p_to_filename(p) + "big2"
data = Data(p, sw_path=sw_path)
data.generate_obs_data(filename=filename, nr_samples=10000)