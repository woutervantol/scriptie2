from imports.data import Data
from imports.params import p
from imports.utility import *

p["resolution"] = 64
p["obs_radius"] = 2 #Mpc
p["nr_uniform_bins_obs_data"] = 20
p["redshift"] = 0.5

p["model"] = "HYDRO_FIDUCIAL"
print("START: ", p["model"])
data = Data(p)
data.generate_obs_data(p_to_filename(p), nr_samples=10000)

# ["HYDRO_WEAK_AGN", "HYDRO_FIDUCIAL", "HYDRO_STRONG_AGN", "HYDRO_STRONGER_AGN", "HYDRO_STRONGEST_AGN", 
#  "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_JETS_published", "HYDRO_STRONG_JETS_published"]
# p["soapfile"] = "halo_properties_0078.hdf5"
# p["simsize"] = "L2800N5040"
# p["snapshot"] = "flamingo_0078/flamingo_0078.hdf5"
# p["model"] = "HYDRO_FIDUCIAL"

