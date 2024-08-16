from imports.data import *
from imports.params import p
from imports.utility import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Which simulation model to use")
args = parser.parse_args()
if args.model:
    p["model"] = args.model


### set parameters
p["resolution"] = 64
p["obs_radius"] = 2 #Mpc
p["nr_uniform_bins_obs_data"] = 20
p["redshift"] = 0.15
p["depth"] = 10 #Mpc

### parameters for large volume simulations
# p["soapfile"] = "halo_properties_0078.hdf5"
# p["simsize"] = "L2800N5040"
# p["snapshot"] = "flamingo_0078/flamingo_0078.hdf5"
# p["model"] = "HYDRO_FIDUCIAL"




### calculate and save mean noise values
gen_base_noise_values(p)


data = Data(p)
### generate and save images. nr_samples is excluding the roll-off so the actual number of images will be less than nr_samples.
data.generate_obs_data(nr_samples=10000)


# ### generate noise for all variations
# for model in simulation_variations:
#     p["model"] = model
#     data = Data(p)
#     filename = p_to_filename(p)

#     ### save noisy data
#     data.load_dataset()
#     noisy_images = data.add_noise(data.images)
#     np.save(p["data_path"] + filename + "_noisy", noisy_images)




