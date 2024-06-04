import h5py
import numpy as np
import swiftsimio as sw
import time
import unyt
from imports.utility import *
from swiftsimio.visualisation import rotation
from astropy.cosmology import z_at_value
import astropy.units as u
import json
from scipy.ndimage import gaussian_filter
from imports.networks import *

class Data():
    def __init__(self, p):
        try:
            self.soap_file = h5py.File(f"{p_to_path(p)}/SOAP/halo_properties_00{int(77-20*p['redshift'])}.hdf5", "r")
        except:
            print(f"could not load {p_to_path(p)}/SOAP/halo_properties_00{int(77-20*p['redshift'])}.hdf5")

        self.sw_path = f"{p_to_path(p)}/{p['snapshot_folder']}/flamingo_00{int(77-20*p['redshift'])}/flamingo_00{int(77-20*p['redshift'])}.hdf5"
        self.selection_type = p["selection_type"]
        self.p = p

        self.properties = []
        # self.nr_halos = self.soap_file[f"{self.selection_type}/CentreOfMass"].shape[0]


    def add_soap_property(self, path):
        self.properties.append(path)

    def make_soap_dataset(self, target_property="DarkMatterMass"):
        ### Old code
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


    def make_nn_dataset(self, filename, target="DarkMatterMass"):
        self.testx, self.testy, self.trainx, self.trainy, self.valx, self.valy, self.mean_x, self.mean_y, self.std_x, self.std_y = load_nn_dataset(self.p)
        self.testx = self.testx[0]
        self.testy = self.testy[0]
        self.trainx = self.trainx[0]
        self.trainy = self.trainy[0]
        self.valx = self.valx[0]
        self.valy = self.valy[0]

    def load_dataset(self, filename):
        ### Load all images, indices and masses
        if self.p["noisy"]:
            self.images = np.load(self.p["data_path"] + filename + "_noisy.npy")
        else:
            self.images = np.load(self.p["data_path"] + filename + ".npy")
        self.indices = np.load(self.p["data_path"] + filename + "_halo_indices.npy")
        self.masses = np.load(self.p["data_path"] + filename + "_masses.npy")



    def load_testset(self, filename):
        ### Load only images, indices and masses from the testset
        if self.p["noisy"]:
            self.images = np.load(self.p["data_path"] + filename + "_noisy.npy")
        else:
            self.images = np.load(self.p["data_path"] + filename + ".npy")
        self.images = self.images[:int(len(self.images)*self.p["test_size"])]
        self.indices = np.load(self.p["data_path"] + filename + "_halo_indices.npy")
        self.indices = self.indices[:int(len(self.indices)*self.p["test_size"])]
        self.masses = np.load(self.p["data_path"] + filename + "_masses.npy")
        self.masses = self.masses[:int(len(self.masses)*self.p["test_size"])]


    def generate_obs_data(self, filename="", nr_samples=100):
        ### Generate images and corresponding soap indices with log uniform masses
        try:
            np.load(self.p["data_path"] + filename + ".npy")
            print(f"File {self.p['data_path'] + filename + '.npy'} already exists.")
            return
        except:
            pass
        flux_dataset = np.array([]) * 1/unyt.s
        time_start = time.time()
        time_last = time.time()

        nr_bins = self.p["nr_uniform_bins_obs_data"]
        mass_bin_edges = np.logspace(13, 15, nr_bins+1)
        halo_indices = self.mass_uniform_halo_indices(mass_bin_edges, nr_samples)
        np.random.shuffle(halo_indices)
        np.save(self.p["data_path"] + filename + "_halo_indices", halo_indices)
        masses = self.soap_file[f"{self.selection_type}/DarkMatterMass"][()][halo_indices]
        np.save(self.p["data_path"] + filename + "_masses", masses)

        flux_ratio, fov = get_flux_ratio(self.p)
        for sample, halo_index in enumerate(halo_indices):
            red_flux, blue_flux = self.make_obs(halo_index, rotate=True)
            fluxes = np.append(red_flux, blue_flux).reshape(1, 2, self.p['resolution'], self.p['resolution'])
            flux_dataset = np.append(flux_dataset, fluxes).reshape(sample+1, 2, self.p['resolution'], self.p['resolution'])
            print(f"Sample {sample} of {len(halo_indices)} done. Time running: {(time.time() - time_start)/60: .2f}m. {time.time() - time_last:.2f}s since last")
            time_last = time.time()

            if sample%100 == 99:
                ### intermediate saving
                np.save(self.p["data_path"] + filename, flux_dataset*flux_ratio*self.p["obs_time"])
                print(f"Saved {self.p['data_path'] + filename}")

        np.save(self.p["data_path"] + filename, flux_dataset*flux_ratio*self.p["obs_time"])

        

    def make_obs(self, halo_index, rotate=False):
            ### make a single projected image of XRay luminosity for a halo
            mask = sw.mask(self.sw_path)
            a = 1/(1+self.p["redshift"])
            position = self.soap_file[f"{self.selection_type}/CentreOfMass"][halo_index] * unyt.Mpc / a
            radius = self.p['obs_radius'] * unyt.Mpc #use a fixed fov
            load_box = [[position[0] - radius, position[0] + radius], 
                        [position[1] - radius, position[1] + radius], 
                        [position[2] - radius, position[2] + radius]]
            mask.constrain_spatial(load_box)
            halo_data = sw.load(self.sw_path, mask=mask)
            current_time = halo_data.metadata.cosmology.lookback_time(self.p["redshift"])
            max_a_allowed = float(1/(1+z_at_value(halo_data.metadata.cosmology.lookback_time, current_time + 15*u.Myr)))
            halo_mask = halo_data.gas.last_agnfeedback_scale_factors < max_a_allowed
            halo_data.gas.red_flux = halo_data.gas.xray_photon_luminosities.erosita_low
            halo_data.gas.blue_flux = halo_data.gas.xray_photon_luminosities.erosita_high
            # print(len(halo_data.gas.red_flux))
            # print(np.sum(halo_data.gas.red_flux == 0))
            # print(halo_data.gas.red_flux[:100])

            if rotate:
                rotation_center = position.copy()
                rotation_center.convert_to_units(unyt.Mpc)
                vector = np.cos(np.random.rand(3)*2*np.pi)
                vector /= np.linalg.norm(vector)
                rotation_matrix = sw.visualisation.rotation.rotation_matrix_from_vector(vector)
            else:
                rotation_center = None
                rotation_matrix = None
            
            red_flux = sw.visualisation.projection.project_gas(
                halo_data,
                resolution=self.p['resolution'], 
                project="red_flux",
                region=[position[0] - radius, position[0] + radius, position[1] - radius, position[1] + radius],
                parallel = True,
                mask = halo_mask,
                rotation_center=rotation_center,
                rotation_matrix=rotation_matrix,
                backend="subsampled"
            )
            blue_flux = sw.visualisation.projection.project_gas(
                halo_data,
                resolution=self.p['resolution'], 
                project="blue_flux", 
                region=[position[0] - radius, position[0] + radius, position[1] - radius, position[1] + radius],
                parallel = True,
                mask = halo_mask,
                rotation_center=rotation_center,
                rotation_matrix=rotation_matrix,
                backend="subsampled"
            )

            ### Convert to float64 since the number is too large for float32
            red_flux = np.float64(red_flux)
            blue_flux = np.float64(blue_flux)
            red_flux.convert_to_units(1/unyt.s /unyt.kpc**2)
            blue_flux.convert_to_units(1/unyt.s /unyt.kpc**2)
            ### pixel value is converted to an integrated flux across the region in space inside the pixel instead of a flux density
            surface_per_pixel = (2*radius/a / self.p['resolution'])**2
            red_flux *= surface_per_pixel
            blue_flux *= surface_per_pixel
            red_flux[np.where(red_flux == 0)] = 1
            blue_flux[np.where(blue_flux == 0)] = 1
            
            return red_flux, blue_flux


    def split_data(self, x, y):
        ### Split the data into test, validation and train set with fractions 0.1:0.2:0.7 
        ### [test : val : train]
        test_split = int(len(x)*self.p["test_size"])
        val_split = test_split + int(len(x)*self.p["val_size"])
        self.testx = x[:test_split]
        self.testy = y[:test_split]
        self.valx = x[test_split:val_split]
        self.valy = y[test_split:val_split]
        self.trainx = x[val_split:]
        self.trainy = y[val_split:]


    def mass_uniform_halo_indices(self, mass_bin_edges, nr_samples):
        ### select halos from a log uniform mass distribution. If 
        nr_bins = len(mass_bin_edges[:-1])
        halos_per_bin = int(nr_samples / nr_bins)
        halo_indices = np.array([], dtype=int)
        masses = self.soap_file[f"{self.selection_type}/DarkMatterMass"][:]
        indices = np.arange(len(masses))
        for i in range(nr_bins):
            bin_indices = indices[np.logical_and(masses > mass_bin_edges[i], masses < mass_bin_edges[i+1])]
            if len(bin_indices) > halos_per_bin:
                choices = np.random.choice(bin_indices, size=(halos_per_bin), replace=False)
            else:
                choices = bin_indices
            halo_indices = np.append(halo_indices, choices)

        return halo_indices


    def add_noise(self, images, noise=True, psf=True):
        filepath = open(self.p['model_path'] + "bgd.json", 'r')
        bgd = json.load(filepath)
        fov = bgd["z0"+str(self.p["redshift"])[2:]]["fov"]
        if noise:
            photon_counts = np.random.poisson(images)
            background_noise = np.ones_like(photon_counts, dtype=float)
            background_noise[:, 0, :, :] *= 1/(64*64)*bgd["bgd_low"]*fov**2*self.p["obs_time"] * self.p["modules"]
            background_noise[:, 1, :, :] *= 1/(64*64)*bgd["bgd_high"]*fov**2*self.p["obs_time"] * self.p["modules"]
            background_noise = np.random.poisson(background_noise)
            images = background_noise + photon_counts
        if psf:
            fwhm = 26 / 60 #arcmin
            images = gaussian_filter(images, 0.6745*fwhm*(64/fov), axes=(2, 3))
        return images