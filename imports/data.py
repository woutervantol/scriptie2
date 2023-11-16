import h5py
import numpy as np
import swiftsimio as sw
import time
import unyt
from imports.utility import *

class Data():
    def __init__(self, p, soap_path="", sw_path = ""):
        self.soap_file = h5py.File(f"{p_to_path(p)}/SOAP/{soap_path}", "r")
        self.sw_path = f"{p_to_path(p)}/snapshots/{sw_path}"
        self.selection_type = p["selection_type"]
        self.p = p
        self.test_size = 0.2
        self.val_size = 0.1

        self.properties = []
        self.nr_halos = self.soap_file[f"{self.selection_type}/CentreOfMass"].shape[0]


    def add_soap_property(self, path):
        self.properties.append(path)

    def make_soap_dataset(self, target_property="TotalMass"):
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
        
    
    def make_obs_dataset(self, filepath, target="TotalMass"):
        data_x = np.load(f"{filepath}.npy")
        indices_y = np.sort(np.load(f"{filepath}_halo_indices.npy"))
        data_y = self.soap_file[f"{self.selection_type}/{target}"][indices_y]
        nonzero = (data_y != 0)

        data_x = data_x[nonzero]
        data_y = data_y[nonzero]

        data_x = np.log10(data_x)
        data_y = np.log10(data_y)
        self.std_x = np.std(data_x, axis=(0, 2, 3))
        self.std_y = np.std(data_y)
        self.mean_x = np.mean(data_x, axis=(0, 2, 3))
        self.mean_y = np.mean(data_y)
        data_x = (data_x - self.mean_x[np.newaxis, :, np.newaxis, np.newaxis]) / self.std_x[np.newaxis, :, np.newaxis, np.newaxis]
        data_y = (data_y - self.mean_y) / self.std_y
        
        data_x, data_y = self.shuffle_data(data_x, data_y)
        self.split_data(data_x, data_y)



    def create_obs_data(self, save_loc="", nr_samples=100):
        res = self.p['resolution']
        fixed_radius = self.p['obs_radius'] * unyt.Mpc
        dataset = np.zeros((nr_samples, 2, res, res))
        time_start = time.time()
        filename = f"obs_data_{p_to_filename(self.p)}_M1e13_rad1Mpc"

        nr_bins = 10
        mass_bin_edges = np.logspace(13, 15, nr_bins+1)
        halo_indices = self.mass_uniform_halo_indices(mass_bin_edges, nr_samples)
        np.save(f"{save_loc}/{filename}_halo_indices", halo_indices)

        for sample in range(len(halo_indices)):
            mask = sw.mask(self.sw_path)
            
            
            position = self.soap_file[f"{self.selection_type}/CentreOfMass"][halo_indices[sample]] * unyt.Mpc
            # radius = self.soap_file[f"{self.selection_type}/SORadius"][halo_indices[sample]] * unyt.Mpc
            radius = fixed_radius
            load_box = [[position[0] - radius, position[0] + radius], 
                        [position[1] - radius, position[1] + radius], 
                        [position[2] - radius, position[2] + radius]]
            mask.constrain_spatial(load_box)
            halo_data = sw.load(self.sw_path, mask=mask)
            halo_data.gas.red_flux = halo_data.gas.xray_luminosities.erosita_low
            halo_data.gas.blue_flux = halo_data.gas.xray_luminosities.erosita_high
            red_flux = sw.visualisation.projection.project_gas(
                halo_data,
                resolution=res, 
                project="red_flux",
                region=[position[0] - radius, position[0] + radius, position[1] - radius, position[1] + radius],
                parallel = True
            )
            blue_flux = sw.visualisation.projection.project_gas(
                halo_data,
                resolution=res, 
                project="blue_flux", 
                region=[position[0] - radius, position[0] + radius, position[1] - radius, position[1] + radius],
                parallel = True
            )
            red_flux = np.float64(red_flux)
            blue_flux = np.float64(blue_flux)
            red_flux.convert_to_units(unyt.erg/unyt.s /unyt.kpc**2)
            blue_flux.convert_to_units(unyt.erg/unyt.s /unyt.kpc**2)
            kpc_per_pixel = (2*radius / res)**2
            red_flux *= kpc_per_pixel
            blue_flux *= kpc_per_pixel

            dataset[sample, 0, :, :] = red_flux
            dataset[sample, 1, :, :] = blue_flux

            print(f"Sample {sample} of {len(halo_indices)} done. Time running: {time.time() - time_start}s")

            if sample%100 == 99:
                #intermediate saving
                np.save(f"{save_loc}/{filename}", dataset)
        
        np.save(f"{save_loc}/{filename}", dataset)



    def split_data(self, x, y):
        test_split = int(len(x)*self.test_size)
        val_split = test_split + int(len(x)*self.val_size)

        self.testx = x[:test_split]
        self.testy = y[:test_split]
        self.valx = x[test_split:val_split]
        self.valy = y[test_split:val_split]
        self.trainx = x[val_split:]
        self.trainy = y[val_split:]


    def shuffle_data(self, x, y):
        shuffled_indices = np.arange(len(x))
        np.random.shuffle(shuffled_indices)
        return x[shuffled_indices], y[shuffled_indices]


    def mass_uniform_halo_indices(self, mass_bin_edges, nr_samples):
        nr_bins = len(mass_bin_edges[:-1])
        halos_per_bin = int(nr_samples / nr_bins)
        halo_indices = np.zeros((halos_per_bin*nr_bins), dtype=int)
        masses = self.soap_file[f"{self.selection_type}/TotalMass"][:]
        indices = np.arange(len(masses))
        for i in range(nr_bins):
            bin_indices = indices[np.logical_and(masses > mass_bin_edges[i], masses < mass_bin_edges[i+1])]
            choices = np.random.choice(bin_indices, size=(halos_per_bin))
            halo_indices[halos_per_bin*i:halos_per_bin*(i+1)] = choices
        
        return halo_indices