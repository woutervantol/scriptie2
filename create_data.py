from imports.data import *
from imports.params import p
from imports.utility import *

### set parameters
p["resolution"] = 64
p["obs_radius"] = 2 #Mpc
p["nr_uniform_bins_obs_data"] = 20
p["redshift"] = 0.15

### parameters for large volume simulations
# p["soapfile"] = "halo_properties_0078.hdf5"
# p["simsize"] = "L2800N5040"
# p["snapshot"] = "flamingo_0078/flamingo_0078.hdf5"
# p["model"] = "HYDRO_FIDUCIAL"

### calculate and save mean noise values
gen_base_noise_values(p)

### generate images for all variations
for model in ["HYDRO_WEAK_AGN", "HYDRO_FIDUCIAL", "HYDRO_STRONG_AGN", "HYDRO_STRONGER_AGN", "HYDRO_STRONGEST_AGN", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_JETS_published", "HYDRO_STRONG_JETS_published"]:
    p["model"] = model
    print("START: ", p["model"])
    data = Data(p)
    filename = p_to_filename(p)
    ### generate and save images. nr_samples is excluding the roll-off so the actual number of images will be less.
    data.generate_obs_data(filename=filename, nr_samples=10000)

    ### save noisy data
    data.load_dataset(filename=filename)
    noisy_images = data.add_noise(data.images)
    np.save(p["data_path"] + filename + "_noisy", noisy_images)




