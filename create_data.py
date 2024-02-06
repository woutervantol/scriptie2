from imports.data import Data
from imports.params import p
from imports.utility import *

p["resolution"] = 64
p["obs_radius"] = 2 #Mpc
p["nr_uniform_bins_obs_data"] = 20

# p["soapfile"] = "halo_properties_0078.hdf5"
# p["simsize"] = "L2800N5040"
# p["snapshot"] = "flamingo_0078/flamingo_0078.hdf5"
p["cosmology"] = "HYDRO_PLANCK"

filename = p_to_filename(p)
data = Data(p)
data.generate_obs_data(filename=filename, nr_samples=10000)
# data.generate_mass_data(filename=filename)

# filename = p_to_filename(p) + "2"
# data = Data(p)
# data.generate_obs_data(filename=filename, nr_samples=10000)

# filename = p_to_filename(p) + "3"
# data = Data(p)
# data.generate_obs_data(filename=filename, nr_samples=10000)

# filename = p_to_filename(p) + "4"
# data = Data(p)
# data.generate_obs_data(filename=filename, nr_samples=10000)

# filename = p_to_filename(p) + "5"
# data = Data(p)
# data.generate_obs_data(filename=filename, nr_samples=10000)