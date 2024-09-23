from imports.data import *
from imports.params import p
from imports.utility import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", help="Which simulation model to use")
args = parser.parse_args()


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


def gen_images(model):
    """generate and save images. nr_samples is excluding the roll-off so the actual number of images will be less than nr_samples."""
    p["model"] = model
    data = Data(p)
    data.generate_linear_data(np.load(p["data_path"] + p_to_filename(p, images=False) + "_halo_indices.npy"))

    try:
        np.load(p["data_path"] + p_to_filename(p)+"_noisy.npy")
        print(p["data_path"] + p_to_filename(p)+"_noisy.npy already exists.")
    except:
        ### generate noise and save noisy dataset if noisy data does not already exist
        data.load_dataset()
        noisy_images = data.add_noise(data.images)
        np.save(p["data_path"] + p_to_filename(p)+"_noisy", noisy_images)

### if simulation variation is given, generate corresponding images. Else generate images from all simulations.
if args.model:
    gen_images(args.model)
else:
    for model in simulation_variations:
        gen_images(model)



