import numpy as np
import swiftsimio as sw


def p_to_path(p):
    return f"{p['flamingo_path']}/{p['simsize']}/{p['model']}"


def p_to_filename(p):
    return f"{p['simsize']}_{p['model']}_{p['selection_type_name']}_res{p['resolution']}_z0{str(p['redshift'])[2:]}"



def get_nth_newest_file(path, n):
    import os
    search_dir = path
    os.chdir(search_dir)
    files = filter(os.path.isfile, os.listdir(search_dir))
    files = [os.path.join(search_dir, f) for f in files] # add path to each file
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-n]


def get_flux_ratio(p):
    z = p["redshift"]
    comoving_distance = 3.08567758e22 * sw.load(f"{p_to_path(p)}/{p['snapshot_folder']}/flamingo_00{int(77-20*p['redshift'])}/flamingo_00{int(77-20*p['redshift'])}.hdf5").metadata.cosmology.comoving_distance(z).value #m
    telescope_surface = np.pi * (p["diameter"]/2)**2 * p["modules"] #m^2
    flux_ratio = telescope_surface /(4*np.pi*(1+z)*comoving_distance**2)#frac of luminosity that arrives at distance r on telescope surface
    angular_distance = comoving_distance / (1+z)
    fov = (np.arcsin(4*3e22/angular_distance)/2/np.pi*360*60) #arcmin
    return flux_ratio, fov




def load_nn_dataset(p, include_self=False, normalise=True):
    filename =  p_to_filename(p)
    testx = []
    testy = []
    valx = []
    valy = []
    trainx = []
    trainy = []
    # indices = []
    radii = []
    for model in ["HYDRO_WEAK_AGN", "HYDRO_FIDUCIAL", "HYDRO_STRONG_AGN", "HYDRO_STRONGER_AGN", "HYDRO_STRONGEST_AGN", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_JETS_published", "HYDRO_STRONG_JETS_published"]:
        if p["simtype"] == "all_but" and model == p["model"] and include_self == False:
            continue
        if p["simtype"] == "single" and model != p["model"]:
            continue
        if p["simtype"] == "extremes" and model != "HYDRO_WEAK_AGN" and model != "HYDRO_STRONGEST_AGN":
            continue
        p_temp = p.copy()
        p_temp["model"] = model
        filename = p_to_filename(p_temp)
        data_x = np.load(p["data_path"] + filename + ".npy") if p["noisy"] == False else np.load(p["data_path"] + filename + "_noisy.npy")
        data_y = np.load(p["data_path"] + filename + "_masses.npy")
        test_split = int(len(data_y)*p["test_size"])
        val_split = test_split + int(len(data_y)*p["val_size"])
        testx.append(np.log10(data_x[:test_split]))
        testy.append(np.log10(data_y[:test_split]))
        valx.append(np.log10(data_x[test_split:val_split]))
        valy.append(np.log10(data_y[test_split:val_split]))
        trainx.append(np.log10(data_x[val_split:]))
        trainy.append(np.log10(data_y[val_split:]))
        if not normalise:
            from imports.data import Data
            data = Data(p_temp)
            ids = np.load(p["data_path"] + filename + "_halo_indices.npy")[val_split:]
            # indices.append(ids)
            radii.append(data.soap_file[f"{data.selection_type}/SORadius"][:][ids] * 1000)



    
    if normalise:
        std_x = np.std(np.concatenate(trainx), axis=(0, 2, 3))
        std_y = np.std(np.concatenate(trainy))
        mean_x = np.mean(np.concatenate(trainx), axis=(0, 2, 3))
        mean_y = np.mean(np.concatenate(trainy))

        for i in range(len(testx)):
            #scale and shift data for better nn training
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
