from imports.data import Data
from imports.params import p
from imports.utility import *

p["resolution"] = 64
p["obs_radius"] = 2 #Mpc
p["nr_uniform_bins_obs_data"] = 50

p["model"] = "HYDRO_PLANCK"

filename = p_to_filename(p)
data = Data(p)
data.generate_obs_data(filename=filename, nr_samples=10000)