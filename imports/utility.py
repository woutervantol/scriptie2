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
    p["model"] = "HYDRO_FIDUCIAL"
    total_bgd = np.loadtxt(p["model_path"]+"bgd.txt", delimiter=",")
    z = p["redshift"]
    H0 = sw.load(f"{p_to_path(p)}/{p['snapshot_folder']}/flamingo_00{int(77-20*p['redshift'])}/flamingo_00{int(77-20*p['redshift'])}.hdf5").metadata.cosmology.H0.value #km/s/Mpc
    c = 3e5 #km/s
    r = c*z/H0 * 3.08567758e22 / (1+z) #m
    telescope_diameter = p["diameter"] #m
    telescope_surface = np.pi * (telescope_diameter/2)**2 * p["modules"] #m^2
    flux_ratio = telescope_surface / (4*np.pi*r**2*(1+z)) #frac of luminosity that arrives at distance r on telescope surface
    fov = (np.arcsin(4*3e22/r)/2/np.pi*360*60) #arcmin^2
    return flux_ratio, fov




def load_nn_dataset(p, shuffled=False):
    filename =  p_to_filename(p)
    # testx = np.empty((0, 2, p["resolution"], p["resolution"]))
    # testy = np.empty((0))
    # valx = np.empty((0, 2, p["resolution"], p["resolution"]))
    # valy = np.empty((0))
    # trainx = np.empty((0, 2, p["resolution"], p["resolution"]))
    # trainy = np.empty((0))
    testx = []
    testy = []
    valx = []
    valy = []
    trainx = []
    trainy = []
    for model in ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_JETS_published", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
        print(f"loading {model}", flush=True)
        if p["simtype"] == "all_but" and model == p["model"]:
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

    # if shuffled:
    #     shuffled_indices = np.arange(len(testy))
    #     np.random.shuffle(shuffled_indices)
    #     testx = testx[shuffled_indices]
    #     testy = testy[shuffled_indices]
    #     shuffled_indices = np.arange(len(valy))
    #     np.random.shuffle(shuffled_indices)
    #     valx = valx[shuffled_indices]
    #     valy = valy[shuffled_indices]
    #     shuffled_indices = np.arange(len(trainy))
    #     np.random.shuffle(shuffled_indices)
    #     trainx = trainx[shuffled_indices]
    #     trainy = trainy[shuffled_indices]

    return testx, testy, trainx, trainy, valx, valy, mean_x, mean_y, std_x, std_y