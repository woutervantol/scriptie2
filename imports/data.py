import h5py
import numpy as np
import swiftsimio as sw
import time
import unyt
from imports.utility import *
from swiftsimio.visualisation import rotation

class Data():
    def __init__(self, p):
        try:
            self.soap_file = h5py.File(f"{p_to_path(p)}/SOAP/{p['soapfile']}", "r")
        except:
            pass
        self.sw_path = f"{p_to_path(p)}/snapshots/{p['snapshot']}"
        self.selection_type = p["selection_type"]
        self.p = p

        self.properties = []
        # self.nr_halos = self.soap_file[f"{self.selection_type}/CentreOfMass"].shape[0]


    def add_soap_property(self, path):
        self.properties.append(path)

    def make_soap_dataset(self, target_property="TotalMass"):
        #old code
        data_x = np.zeros((self.nr_halos, len(self.properties)))
        for i in range(len(self.properties)):
            data_x[:,i] = self.soap_file[f"{self.selection_type}{self.properties[i]}"]
        data_y = np.array(self.soap_file[f"{self.selection_type}{target_property}"])

        nonzero_target = (data_y != 0)
        nonzero_data = (np.sum(data_x == 0, axis=1) == 0)
        nonzero = nonzero_target * nonzero_data

        data_x = data_x[nonzero]
        data_y = data_y[nonzero]

        data_x = np.log10(data_x)
        data_y = np.log10(data_y)
        self.std_x = np.std(data_x, axis=0)
        self.std_y = np.std(data_y)
        self.mean_x = np.mean(data_x, axis=0)
        self.mean_y = np.mean(data_y)
        data_x = (data_x - self.mean_x) / self.std_x
        data_y = (data_y - self.mean_y) / self.std_y
        
        data_x, data_y = self.shuffle_data(data_x, data_y)
        self.split_data(data_x, data_y)


    def make_nn_dataset(self, filename, target="TotalMass"):
        data_x = np.load(self.p["data_path"] + filename + ".npy")
        # indices_y = np.load(self.p["data_path"] + filename + "_halo_indices.npy")
        # data_y = self.soap_file[f"{self.selection_type}/{target}"][()][indices_y]
        data_y = np.load(self.p["data_path"] + filename + "_masses.npy")

        data_x = np.log10(data_x)
        data_y = np.log10(data_y)
        self.std_x = np.std(data_x, axis=(0, 2, 3))
        self.std_y = np.std(data_y)
        self.mean_x = np.mean(data_x, axis=(0, 2, 3))
        self.mean_y = np.mean(data_y)

        #scale and shift data for better nn training
        data_x = (data_x - self.mean_x[np.newaxis, :, np.newaxis, np.newaxis]) / self.std_x[np.newaxis, :, np.newaxis, np.newaxis]
        data_y = (data_y - self.mean_y) / self.std_y
        
        #Select the correct image for single channel runs
        if self.p["channel"]=="low": 
            data_x = data_x[:,:1,:,:]
        elif self.p["channel"]=="high": 
            data_x = data_x[:,1:,:,:]
        else:
            pass

        self.split_data(data_x, data_y)

    def load_dataset(self, filename):
        #load all images, indices and masses
        self.images = np.load(self.p["data_path"] + filename + ".npy")
        self.indices = np.load(self.p["data_path"] + filename + "_halo_indices.npy")
        self.masses = self.soap_file[f"{self.selection_type}/TotalMass"][()][self.indices]
    

    def load_testset(self, filename):
        #load only images, indices and masses from the testset
        self.images = np.load(self.p["data_path"] + filename + ".npy")
        self.images = self.images[:int(len(self.images)*self.p["test_size"])]
        self.indices = np.load(self.p["data_path"] + filename + "_halo_indices.npy")
        self.indices = self.indices[:int(len(self.indices)*self.p["test_size"])]
        self.masses = self.soap_file[f"{self.selection_type}/TotalMass"][()][self.indices]


    def generate_obs_data(self, filename="", nr_samples=100):
        #Generate images and corresponding soap indices with log uniform masses
        dataset = np.array([]) * unyt.erg/unyt.s
        time_start = time.time()
        time_last = time.time()

        nr_bins = self.p["nr_uniform_bins_obs_data"]
        mass_bin_edges = np.logspace(13, 15, nr_bins+1)
        halo_indices = self.mass_uniform_halo_indices(mass_bin_edges, nr_samples)
        np.random.shuffle(halo_indices)

        np.save(self.p["data_path"] + filename + "_halo_indices", halo_indices)

        for sample, halo_index in enumerate(halo_indices):
            red_flux, blue_flux = self.make_obs(halo_index)
            fluxes = np.append(red_flux, blue_flux).reshape(1, 2, self.p['resolution'], self.p['resolution'])
            dataset = np.append(dataset, fluxes).reshape(sample+1, 2, self.p['resolution'], self.p['resolution'])
            print(f"Sample {sample} of {len(halo_indices)} done. Time running: {(time.time() - time_start)/60: .2f}m. {time.time() - time_last:.2f}s since last")
            time_last = time.time()

            if sample%100 == 99:
                #intermediate saving
                np.save(self.p["data_path"] + filename, dataset)
                print(f"Saved {self.p['data_path'] + filename}")
        
        np.save(self.p["data_path"] + filename, dataset)

    def make_obs(self, halo_index, rotate=False):
            #make a single projected image of XRay luminosity for a halo
            mask = sw.mask(self.sw_path)
            position = self.soap_file[f"{self.selection_type}/CentreOfMass"][halo_index] * unyt.Mpc
            radius = self.p['obs_radius'] * unyt.Mpc #use a fixed "distance" to the object, which 
            load_box = [[position[0] - radius, position[0] + radius], 
                        [position[1] - radius, position[1] + radius], 
                        [position[2] - radius, position[2] + radius]]
            mask.constrain_spatial(load_box)

            halo_data = sw.load(self.sw_path, mask=mask)
            halo_mask = halo_data.gas.last_agnfeedback_scale_factors < 0.995
            halo_data.gas.red_flux = halo_data.gas.xray_luminosities.erosita_low
            halo_data.gas.blue_flux = halo_data.gas.xray_luminosities.erosita_high


            red_flux = sw.visualisation.projection.project_gas(
                halo_data,
                resolution=self.p['resolution'], 
                project="red_flux",
                region=[position[0] - radius, position[0] + radius, position[1] - radius, position[1] + radius],
                parallel = True,
                mask = halo_mask
            )
            blue_flux = sw.visualisation.projection.project_gas(
                halo_data,
                resolution=self.p['resolution'], 
                project="blue_flux", 
                region=[position[0] - radius, position[0] + radius, position[1] - radius, position[1] + radius],
                parallel = True,
                mask = halo_mask
            )
            ### retry rotating the object randomly until the brightest pixel is in the middel (to prevent bright pixels from agn injection)
            # retries = 0
            # brightest_pixel = np.unravel_index(np.argmax(blue_flux), blue_flux.shape)
            # while brightest_pixel not in np.array([[31, 31], [31, 32], [32, 31], [32, 32]]):
            #     retries += 1
            #     rotation_center = position.copy()
            #     rotation_center.convert_to_units(unyt.Mpc)
            #     vector = np.cos(np.random.rand(3)*np.pi)
            #     vector /= np.linalg.norm(vector)
            #     rotation_matrix = sw.visualisation.rotation.rotation_matrix_from_vector(vector)
            #     red_flux = sw.visualisation.projection.project_gas(
            #         halo_data,
            #         resolution=self.p['resolution'], 
            #         project="red_flux",
            #         region=[position[0] - radius, position[0] + radius, position[1] - radius, position[1] + radius],
            #         parallel = True,
            #         mask = halo_mask,
            #         rotation_center=rotation_center,
            #         rotation_matrix=rotation_matrix
            #     )
            #     blue_flux = sw.visualisation.projection.project_gas(
            #         halo_data,
            #         resolution=self.p['resolution'], 
            #         project="blue_flux", 
            #         region=[position[0] - radius, position[0] + radius, position[1] - radius, position[1] + radius],
            #         parallel = True,
            #         mask = halo_mask,
            #         rotation_center=rotation_center,
            #         rotation_matrix=rotation_matrix
            #     )
            #     brightest_pixel = np.unravel_index(np.argmax(blue_flux), blue_flux.shape)

            #Convert to float64 since the number is too large for float32
            red_flux = np.float64(red_flux)
            blue_flux = np.float64(blue_flux)
            red_flux.convert_to_units(unyt.erg/unyt.s /unyt.kpc**2)
            blue_flux.convert_to_units(unyt.erg/unyt.s /unyt.kpc**2)
            #pixel value is converted to an integrated flux across the region in space inside the pixel instead of a flux density
            kpc_per_pixel = (2*radius / self.p['resolution'])**2
            red_flux *= kpc_per_pixel
            blue_flux *= kpc_per_pixel
            # if retries != 0:
            #     print(f"Tried {retries+1} times for a correct angle")
            return red_flux, blue_flux

    def split_data(self, x, y):
        #Split the data into test, validation and train set with fractions 0.1:0.2:0.7 
        #[test : val : train]
        test_split = int(len(x)*self.p["test_size"])
        val_split = test_split + int(len(x)*self.p["val_size"])
        self.testx = x[:test_split]
        self.testy = y[:test_split]
        self.valx = x[test_split:val_split]
        self.valy = y[test_split:val_split]
        self.trainx = x[val_split:]
        self.trainy = y[val_split:]


    def shuffle_data(self, x, y):
        #old code
        shuffled_indices = np.arange(len(x))
        np.random.shuffle(shuffled_indices)
        return x[shuffled_indices], y[shuffled_indices]


    def mass_uniform_halo_indices(self, mass_bin_edges, nr_samples):
        #select halos from a log uniform mass distribution. If 
        nr_bins = len(mass_bin_edges[:-1])
        halos_per_bin = int(nr_samples / nr_bins)
        halo_indices = np.array([], dtype=int)
        masses = self.soap_file[f"{self.selection_type}/TotalMass"][:]
        indices = np.arange(len(masses))
        for i in range(nr_bins):
            bin_indices = indices[np.logical_and(masses > mass_bin_edges[i], masses < mass_bin_edges[i+1])]
            if len(bin_indices) > halos_per_bin:
                choices = np.random.choice(bin_indices, size=(halos_per_bin), replace=False)
            else:
                choices = bin_indices
            halo_indices = np.append(halo_indices, choices)

        return halo_indices