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
from imports.xray_interpolator import XrayCalculator

class Data():
    def __init__(self, p):
        try:
            self.soap_file = h5py.File(f"{p_to_path(p)}/SOAP/halo_properties_00{int(77-20*p['redshift'])}.hdf5", "r")
        except:
            print(f"could not load {p_to_path(p)}/SOAP/halo_properties_00{int(77-20*p['redshift'])}.hdf5")

        self.sw_path = f"{p_to_path(p)}/{p['snapshot_folder']}/flamingo_00{int(77-20*p['redshift'])}/flamingo_00{int(77-20*p['redshift'])}.hdf5"
        self.selection_type = p["selection_type"]
        self.p = p

    def make_nn_dataset(self, target="DarkMatterMass"):
        self.testx, self.testy, self.trainx, self.trainy, self.valx, self.valy, self.mean_x, self.mean_y, self.std_x, self.std_y = load_nn_dataset(self.p)
        self.testx = self.testx[0]
        self.testy = self.testy[0]
        self.trainx = self.trainx[0]
        self.trainy = self.trainy[0]
        self.valx = self.valx[0]
        self.valy = self.valy[0]

    def load_dataset(self):
        """Load all images, indices and masses"""
        self.images = np.load(self.p["data_path"] + p_to_filename(self.p) + ".npy")
        self.indices = np.load(self.p["data_path"] + p_to_filename(self.p, images=False) + "_halo_indices.npy")
        self.masses = np.load(self.p["data_path"] + p_to_filename(self.p, images=False) + "_masses.npy")



    def load_testset(self):
        """Load only the testset images, indices and masses"""
        self.images = np.load(self.p["data_path"] + p_to_filename(self.p) + ".npy")
        self.images = self.images[:int(len(self.images)*self.p["test_size"])]
        self.indices = np.load(self.p["data_path"] + p_to_filename(self.p, images=False) + "_halo_indices.npy")
        self.indices = self.indices[:int(len(self.indices)*self.p["test_size"])]
        self.masses = np.load(self.p["data_path"] + p_to_filename(self.p, images=False) + "_masses.npy")
        self.masses = self.masses[:int(len(self.masses)*self.p["test_size"])]

    def generate_linear_data(self, halo_indices):
        filename = p_to_filename(self.p) + "_linear"
        try:
            np.load(self.p["data_path"] + filename + ".npy")
            print(filename+".npy already exists.")
            return
        except:
            pass
        time_start = time.time()
        time_last = time.time()
        flux_ratio, fov = get_flux_ratio(self.p)
        flux_ratio = flux_ratio * (1/unyt.m**2)
        flux_dataset = np.array([]) * unyt.cm**2/unyt.s
        for sample, halo_index in enumerate(halo_indices):
            ### make the image and add it to the dataset
            red_flux, blue_flux = self.make_obs(halo_index, rotate=False, remove_substructures=True)
            fluxes = np.append(red_flux, blue_flux).reshape(1, 2, self.p['resolution'], self.p['resolution'])
            flux_dataset = np.append(flux_dataset, fluxes).reshape(sample+1, 2, self.p['resolution'], self.p['resolution'])
            print(f"Sample {sample} of {len(halo_indices)} done. Time running: {(time.time() - time_start)/60: .2f}m. {time.time() - time_last:.2f}s since last")
            time_last = time.time()

            if sample%100 == 99:
                ### intermediate saving
                np.save(self.p["data_path"] + filename, flux_dataset*flux_ratio*self.p["obs_time"])
                print(f"Saved {self.p['data_path'] + filename}")
        np.save(self.p["data_path"] + filename, flux_dataset*flux_ratio*self.p["obs_time"])


    def generate_obs_data(self, nr_samples=100):
        """Generate images with log uniform mass distribution"""
        ### check if the file already exists
        filename = p_to_filename(self.p)
        try:
            np.load(self.p["data_path"] + filename + ".npy")
            print(f"File {self.p['data_path'] + filename + '.npy'} already exists.")
            return
        except:
            pass
        flux_dataset = np.array([]) * unyt.cm**2/unyt.s
        time_start = time.time()

        ### choose indices for halos for log uniform distribution with roll-off
        nr_bins = self.p["nr_uniform_bins_obs_data"]
        mass_bin_edges = np.logspace(13, 15, nr_bins+1)
        halo_indices = self.mass_uniform_halo_indices(mass_bin_edges, nr_samples)

        np.random.shuffle(halo_indices)
        np.save(self.p["data_path"] + filename + "_halo_indices", halo_indices)
        masses = self.soap_file[f"{self.selection_type}/DarkMatterMass"][()][halo_indices]
        np.save(self.p["data_path"] + filename + "_masses", masses)

        flux_ratio, fov = get_flux_ratio(self.p)
        flux_ratio = flux_ratio * (1/unyt.m**2)
        for sample, halo_index in enumerate(halo_indices):
            ### make the image and add it to the dataset
            red_flux, blue_flux = self.make_obs(halo_index, rotate=False)
            fluxes = np.append(red_flux, blue_flux).reshape(1, 2, self.p['resolution'], self.p['resolution'])
            flux_dataset = np.append(flux_dataset, fluxes).reshape(sample+1, 2, self.p['resolution'], self.p['resolution'])
            print(f"Sample {sample} of {len(halo_indices)} done. Time running: {(time.time() - time_start)/60: .2f}m. {time.time() - time_last:.2f}s since last")
            time_last = time.time()

            if sample%100 == 99:
                ### intermediate saving
                np.save(self.p["data_path"] + filename, flux_dataset*flux_ratio*self.p["obs_time"])
                print(f"Saved {self.p['data_path'] + filename}")

        np.save(self.p["data_path"] + filename, flux_dataset*flux_ratio*self.p["obs_time"])

        

    def make_obs(self, halo_index, rotate=False, remove_substructures=False, interpolation=True):
        """Make single projected image of photon luminosity of a halo with given flamingo index"""
        mask = sw.mask(self.sw_path)
        a = 1/(1+self.p["redshift"])
        position = self.soap_file[f"{self.selection_type}/CentreOfMass"][halo_index] * unyt.Mpc / a
        radius = self.p['obs_radius'] * unyt.Mpc #use a fixed volume
        depth = self.p['obs_depth'] * unyt.Mpc #use a fixed depth
        ### load all particles in a square box with sides of length 2*radius
        load_box = [[position[0] - radius, position[0] + radius], 
                    [position[1] - radius, position[1] + radius], 
                    [position[2] - depth, position[2] + depth]]
        mask.constrain_spatial(load_box)
        halo_data = sw.load(self.sw_path, mask=mask)
        current_time = halo_data.metadata.cosmology.lookback_time(self.p["redshift"])
        
        ### filter out particles with AGN feedback more recent than 15 Myr
        max_a_allowed = float(1/(1+z_at_value(halo_data.metadata.cosmology.lookback_time, current_time + 15*u.Myr)))
        halo_mask = halo_data.gas.last_agnfeedback_scale_factors < max_a_allowed

        if remove_substructures:
            fof_group = halo_data.gas.fofgroup_ids[np.argmin(np.linalg.norm(halo_data.gas.coordinates - position, axis=1))]
            bound_particles_mask = (halo_data.gas.fofgroup_ids == fof_group) | (halo_data.gas.fofgroup_ids == 2147483647)
            # bound_particles_mask = halo_data.gas.fofgroup_ids == fof_group
            # bound_particles_mask = halo_data.gas.fofgroup_ids == 2147483647
            halo_mask = halo_mask * bound_particles_mask

        halo_data.gas.red_flux = halo_data.gas.xray_photon_luminosities.erosita_low
        halo_data.gas.blue_flux = halo_data.gas.xray_photon_luminosities.erosita_high

        ### project to a random direction
        if rotate:
            rotation_center = position.copy()
            rotation_center.convert_to_units(unyt.Mpc)
            vector = np.cos(np.random.rand(3)*2*np.pi)
            vector /= np.linalg.norm(vector)
            rotation_matrix = sw.visualisation.rotation.rotation_matrix_from_vector(vector)
        else:
            rotation_center = None
            rotation_matrix = None

        if interpolation:
            ### interpolate luminosities in the broad band
            mass_fraction = np.zeros((len(halo_data.gas.smoothed_element_mass_fractions.hydrogen), 9))
            mass_fraction[:, 0] = halo_data.gas.smoothed_element_mass_fractions.hydrogen
            mass_fraction[:, 1] = halo_data.gas.smoothed_element_mass_fractions.helium
            mass_fraction[:, 2] = halo_data.gas.smoothed_element_mass_fractions.carbon
            mass_fraction[:, 3] = halo_data.gas.smoothed_element_mass_fractions.nitrogen
            mass_fraction[:, 4] = halo_data.gas.smoothed_element_mass_fractions.oxygen
            mass_fraction[:, 5] = halo_data.gas.smoothed_element_mass_fractions.neon
            mass_fraction[:, 6] = halo_data.gas.smoothed_element_mass_fractions.magnesium
            mass_fraction[:, 7] = halo_data.gas.smoothed_element_mass_fractions.silicon
            mass_fraction[:, 8] = halo_data.gas.smoothed_element_mass_fractions.iron

            table_name = '/net/dodder/data2/braspenning/X_Ray_table_metals_full_withconvolved.hdf5'

            xray_calc = XrayCalculator(self.p["redshift"], table_name, ['erosita-low'], ['photons_convolved'])
            idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, abundance_to_solar, joint_mask, volumes, data_n = xray_calc.find_indices(halo_data.gas.densities, halo_data.gas.temperatures, mass_fraction, halo_data.gas.masses, fill_value = 0)
            interpolated_quantities = xray_calc.interpolate_X_Ray(idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, abundance_to_solar, joint_mask, volumes, data_n, bands = ['erosita-low'], observing_types = ['photons_convolved'], fill_value = 0)
            halo_data.gas.red_flux = sw.cosmo_array(interpolated_quantities.flatten(), cosmo_factor=halo_data.gas.red_flux.cosmo_factor)

            xray_calc = XrayCalculator(self.p["redshift"], table_name, ['erosita-high'], ['photons_convolved'])
            idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, abundance_to_solar, joint_mask, volumes, data_n = xray_calc.find_indices(halo_data.gas.densities, halo_data.gas.temperatures, mass_fraction, halo_data.gas.masses, fill_value = 0)
            interpolated_quantities = xray_calc.interpolate_X_Ray(idx_he, idx_T, idx_n, t_z, d_z, t_T, d_T, t_nH, d_nH, t_He, d_He, abundance_to_solar, joint_mask, volumes, data_n, bands = ['erosita-high'], observing_types = ['photons_convolved'], fill_value = 0)
            
            halo_data.gas.blue_flux = sw.cosmo_array(interpolated_quantities.flatten(), cosmo_factor=halo_data.gas.blue_flux.cosmo_factor)

        ### perform the projections in both broad bands
        red_flux = sw.visualisation.projection.project_gas(
            halo_data,
            resolution=self.p['resolution'], 
            project="red_flux",
            region=[position[0] - radius, position[0] + radius, position[1] - radius, position[1] + radius, position[2]-depth, position[2]+depth],
            parallel = True,
            mask = halo_mask,
            rotation_center=rotation_center,
            rotation_matrix=rotation_matrix,
            backend="subsampled" #saves as double precision float, which is needed to prevent overflow
        )
        blue_flux = sw.visualisation.projection.project_gas(
            halo_data,
            resolution=self.p['resolution'], 
            project="blue_flux", 
            region=[position[0] - radius, position[0] + radius, position[1] - radius, position[1] + radius, position[2]-depth, position[2]+depth],
            parallel = True,
            mask = halo_mask,
            rotation_center=rotation_center,
            rotation_matrix=rotation_matrix,
            backend="subsampled"
        )

        ### convert surface brightness to total photons emitted by the area of the pixel in photons*cm**2/s, where cm**2 is effective receiving surface area
        surface_per_pixel = (2*radius/a / self.p['resolution'])**2
        red_flux = red_flux*surface_per_pixel
        blue_flux = blue_flux*surface_per_pixel
        ### set emitted photons per second to 1 if there are no particles in a region. prevents errors when taking log
        red_flux[np.where(red_flux == 0)] = 1
        blue_flux[np.where(blue_flux == 0)] = 1
        
        return red_flux, blue_flux # cm**2/s if interpolation, else 1/s


    def split_data(self, x, y):
        """Splits the data into test, validation and train set with fractions 0.1:0.2:0.7"""
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
        """select halos from a log uniform mass distribution."""
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
        """Returns the 4 dimensional input images [image_nr, channel, pixel value, pixel value], 
        including shot noise, instrument noise, background noise and optionally PSF convolution."""
        filepath = open(self.p['model_path'] + "bgd.json", 'r')
        bgd = json.load(filepath)
        fov = bgd["z0"+str(self.p["redshift"])[2:]]["fov"]
        if noise:
            ### shot noise
            photon_counts = np.random.poisson(images)

            ### set mean value and sample from poisson with that mean for background
            background_noise = np.ones_like(photon_counts, dtype=float)
            background_noise[:, 0, :, :] *= 1/(64*64)*bgd["bgd_low"]*fov**2*self.p["obs_time"] * self.p["modules"]
            background_noise[:, 1, :, :] *= 1/(64*64)*bgd["bgd_high"]*fov**2*self.p["obs_time"] * self.p["modules"]
            background_noise = np.random.poisson(background_noise)

            images = background_noise + photon_counts
        if psf:
            ### half energy width of 26'' from eROSITA
            hew = 26 / 60 #arcmin
            ### 1 sigma = 1/0.6745 hew. Translated to pixels
            images = gaussian_filter(images.astype(np.float64), 1/0.6745 *hew*(64/fov), axes=(2, 3))
        return images


