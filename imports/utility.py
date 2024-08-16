import numpy as np
import swiftsimio as sw

### define order of simulation variations globally
simulation_variations = ["HYDRO_WEAK_AGN", "HYDRO_FIDUCIAL", "HYDRO_STRONG_AGN", "HYDRO_STRONGER_AGN", "HYDRO_STRONGEST_AGN", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_JETS_published", "HYDRO_STRONG_JETS_published"]
colors = ['#abd0e6', '#117733','#6aaed6','#3787c0','#105ba4', '#FF8C40','#CC4314','#7EFF4B','#55E18E']
names = [r"+2$\sigma$ fgas", "Fiducial", r"-2$\sigma$ fgas", r"-4$\sigma$ fgas", r"-8$\sigma$ fgas", r"M*-$\sigma$", r"M*-$\sigma$_fgas-4$\sigma$", r"Jet", r"Jet_fgas-4$\sigma$"]



def p_to_path(p):
    return f"{p['flamingo_path']}/{p['simsize']}/{p['model']}"


def p_to_filename(p, data=False):
    ### when loading datasets, addons are not used. This distinction is necessary during testing 
    name_addon = ""
    if not data:
        if p["simtype"] != "single":
            name_addon += "_"+p["simtype"]
        if p["noisy"] == True:
            name_addon += "_"+"noisy"
        if p["channel"] != "2chan":
            name_addon += "_"+p["channel"]
    return f"{p['simsize']}_{p['model']}_z0{str(p['redshift'])[2:]}" + name_addon



def get_flux_ratio(p):
    """Returns the fraction of emitted photons that arrives at the telescope, and the field of view along one axis in arcmin (it's a square image)"""
    z = p["redshift"]
    comoving_distance = 3.08567758e22 * sw.load(f"{p_to_path(p)}/{p['snapshot_folder']}/flamingo_00{int(77-20*p['redshift'])}/flamingo_00{int(77-20*p['redshift'])}.hdf5").metadata.cosmology.comoving_distance(z).value #m
    # telescope_surface = np.pi * (p["diameter"]/2)**2 * p["modules"] #m^2
    
    ### photon flux equation with added factor 1/(1+z)^2 due to expansion of space and (1+z) due to wavelength dilation
    ### The effective telescope surface is wavelength dependent, and has already been included during generation, so is omitted here
    flux_ratio = 1 /(4*np.pi*(1+z)*comoving_distance**2)#frac of luminosity that arrives at distance r on telescope surface
    angular_distance = comoving_distance / (1+z)
    fov = (np.arcsin(4*3.08567758e22/angular_distance)/2/np.pi*360*60) #arcmin
    return flux_ratio, fov #flux ratio is in 1/m**2


def gen_base_noise_values(p):
    """Generates the mean background noise flux by integrating Figure 13 from Predehl et al, 2021.
    Also saves field of view and ratio between sent and received flux."""
    import json
    ### Check if file already exists
    try:
        with open(p['model_path'] + "bgd.json", 'r') as filepath:
            json.load(filepath)
        print(f"File {p['model_path'] + 'bgd.json'} already exists")
        return
    except:
        pass
    p_temp = p.copy()
    bgd_dict = {}
    total_bgd = np.loadtxt(p_temp["model_path"]+"bgd.txt", delimiter=",")
    bgd_low = total_bgd[total_bgd[:,0] < 2.3]
    bgd_high = total_bgd[total_bgd[:,0] >= 2.3]
    bgd_low = np.trapz(bgd_low[:,1], bgd_low[:,0]) * p_temp["modules"]
    bgd_high = np.trapz(bgd_high[:,1], bgd_high[:,0]) * p_temp["modules"]
    bgd_dict["bgd_low"] = bgd_low #counts / s / arcmin^2
    bgd_dict["bgd_high"] = bgd_high

    bgd_dict["z015"] = {}
    p_temp["redshift"] = 0.15
    p_temp["model"] = "HYDRO_FIDUCIAL"
    
    flux_ratio, fov = get_flux_ratio(p_temp)
    bgd_dict["z015"]["flux_ratio"] = flux_ratio
    bgd_dict["z015"]["fov"] = fov

    bgd_dict["z05"] = {}
    p_temp["redshift"] = 0.5
    p_temp["model"] = "HYDRO_FIDUCIAL"
    
    flux_ratio, fov = get_flux_ratio(p_temp)
    bgd_dict["z05"]["flux_ratio"] = flux_ratio.tolist()
    bgd_dict["z05"]["fov"] = fov

    
    with open(p_temp['model_path'] + "bgd.json", 'w') as filepath:
        json.dump(bgd_dict, filepath, indent=4)


def load_nn_dataset(p, include_self=False, normalise=True):
    """Returns normalized dataset for use by neural networks. 
    Returns a list with each element containing images for one simulation variation. 
    
    Note: normalised=False returns the unnormalised training sets of the given simulation variations for fitting the linear model."""
    filename =  p_to_filename(p)
    testx = []
    testy = []
    valx = []
    valy = []
    trainx = []
    trainy = []
    radii = []
    for model in simulation_variations:
        ### skip the variations we do not want to have in the dataset
        if p["simtype"] == "all_but" and model == p["model"] and include_self == False:
            continue
        if p["simtype"] == "single" and model != p["model"]:
            continue
        if p["simtype"] == "extremes" and model != "HYDRO_WEAK_AGN" and model != "HYDRO_STRONGEST_AGN":
            continue

        ### Take basic parameters for loading datafiles
        p_temp = p.copy()
        p_temp["model"] = model
        p_temp["simtype"] = "single"
        p_temp["noisy"] = False
        p_temp["channel"] = "2chan"
        filename = p_to_filename(p_temp)
        data_x = np.load(p["data_path"] + filename + ".npy") if p["noisy"] == False else np.load(p["data_path"] + filename + "_noisy.npy")
        data_y = np.load(p["data_path"] + filename + "_masses.npy")
        test_split = int(len(data_y)*p["test_size"])
        val_split = test_split + int(len(data_y)*p["val_size"])
        ### save the log10 of the pixel values
        testx.append(np.log10(data_x[:test_split]))
        testy.append(np.log10(data_y[:test_split]))
        valx.append(np.log10(data_x[test_split:val_split]))
        valy.append(np.log10(data_y[test_split:val_split]))
        trainx.append(np.log10(data_x[val_split:]))
        trainy.append(np.log10(data_y[val_split:]))
        ### for fitting the linear model, the r_500 values are also needed
        if not normalise:
            from imports.data import Data
            data = Data(p_temp)
            ids = np.load(p["data_path"] + filename + "_halo_indices.npy")[val_split:]
            # indices.append(ids)
            radii.append(data.soap_file[f"{data.selection_type}/SORadius"][:][ids] * 1000)



    
    if normalise:
        ### take the std and mean of all pixels in all images in the dataset, for each energy band seperately.
        std_x = np.std(np.concatenate(trainx), axis=(0, 2, 3))
        std_y = np.std(np.concatenate(trainy))
        mean_x = np.mean(np.concatenate(trainx), axis=(0, 2, 3))
        mean_y = np.mean(np.concatenate(trainy))

        for i in range(len(testx)):
            ### scale and shift data for better nn training
            testx[i] = (testx[i] - mean_x[np.newaxis, :, np.newaxis, np.newaxis]) / std_x[np.newaxis, :, np.newaxis, np.newaxis]
            testy[i] = (testy[i] - mean_y) / std_y
            valx[i] = (valx[i] - mean_x[np.newaxis, :, np.newaxis, np.newaxis]) / std_x[np.newaxis, :, np.newaxis, np.newaxis]
            valy[i] = (valy[i] - mean_y) / std_y
            trainx[i] = (trainx[i] - mean_x[np.newaxis, :, np.newaxis, np.newaxis]) / std_x[np.newaxis, :, np.newaxis, np.newaxis]
            trainy[i] = (trainy[i] - mean_y) / std_y
    
    
    for i in range(len(testx)):
        #Select the correct image for single channel runs
        if p["channel"]=="low": 
            testx[i] = testx[i][:,:1,:,:]
            valx[i] = valx[i][:,:1,:,:]
            trainx[i] = trainx[i][:,:1,:,:]
        elif p["channel"]=="high": 
            testx[i] = testx[i][:,1:,:,:]
            valx[i] = valx[i][:,1:,:,:]
            trainx[i] = trainx[i][:,1:,:,:]
        else:
            pass

    if normalise:
        return testx, testy, trainx, trainy, valx, valy, mean_x, mean_y, std_x, std_y
    else:
        return np.concatenate(trainx), np.concatenate(trainy), np.concatenate(radii)
