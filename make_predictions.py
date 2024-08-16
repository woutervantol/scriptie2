from imports.networks import *
from imports.data import *
from imports.params import p
import json


### Linear model is same as in scaling_relation.py
def powerlaw2d(xy, L0_low, p1_low, p2_low, L0_high, p1_high, p2_high, M0):
    """Returns mass prediction for 2xN array of luminosities xy, given fit parameters"""
    x, y = xy
    fit_low = np.zeros(len(x))
    ### prediction along the low energy direction
    fit_low[x < L0_low] = p1_low * (x[x < L0_low] - L0_low)
    fit_low[x >= L0_low] = p2_low * (x[x >= L0_low] - L0_low)
    fit_high = np.zeros(len(y))
    ### prediction along the high energy direction
    fit_high[y < L0_high] = p1_high * (y[y < L0_high] - L0_high)
    fit_high[y >= L0_high] = p2_high * (y[y >= L0_high] - L0_high)
    return fit_low + fit_high + M0



def predict_with_scaling_relations(p):
    """Returns predictions with linear model on testset"""
    ### load background values to subtract them from noisy images
    bgd = json.load(open(p["model_path"]+"bgd.json", "r"))
    bgd_low = bgd["bgd_low"] * bgd["z0"+str(p["redshift"])[2:]]["fov"]**2 * p["obs_time"] /64/64 * p["modules"] #counts / pixel
    bgd_high = bgd["bgd_high"] * bgd["z0"+str(p["redshift"])[2:]]["fov"]**2 * p["obs_time"] /64/64 * p["modules"]
    predictions = []
    for testmodel in simulation_variations:
        ### load images in testset
        p_testset = p.copy()
        p_testset["model"] = testmodel
        if p_testset["noisy"]:
            ### workaround since old convolution was wrong, so recalculate noise here
            p_testset["noisy"] = False
            data = Data(p_testset)
            data.load_testset()
            data.images = data.add_noise(data.images)
            p_testset["noisy"] = True
        else:
            data = Data(p_testset)
            data.load_testset()

        ### add pixel values within r_500
        pixel_size = 4 * 1000 / 64  #4Mpc * 1000kpc/Mpc / 64pixels
        points_low = []
        points_high = []
        for i, haloindex in enumerate(data.indices):
            r = data.soap_file[f"{data.selection_type}/SORadius"][haloindex] * 1000 #kpc
            radius = r / pixel_size
            X, Y = np.ogrid[:p['resolution'], :p['resolution']]
            dist_from_center = np.sqrt((X + 0.5-int(p['resolution']/2))**2+(Y+0.5-int(p['resolution']/2))**2)
            mask_circ = dist_from_center<=radius
            if p["noisy"]:
                ### do not allow negative values for photon counts
                luminosity_circle_low = np.log10(max(np.sum(data.images[i, 0][mask_circ] - bgd_low), 1))
                luminosity_circle_high = np.log10(max(np.sum(data.images[i, 1][mask_circ] - bgd_high), 1))
            else:
                luminosity_circle_low = np.log10(np.sum(data.images[i, 0][mask_circ]))
                luminosity_circle_high = np.log10(np.sum(data.images[i, 1][mask_circ]))

            points_low.append(luminosity_circle_low)
            points_high.append(luminosity_circle_high)
        
        ### make prediction using parameters found by scaling_relation.py
        if p["simtype"] == "single" or p["simtype"] == "all_but":
            coeffs = json.load(open(p["model_path"]+"powerlawcoeffs.json", "r"))[p["simtype"]][p["model"]]
        else:
            coeffs = json.load(open(p["model_path"]+"powerlawcoeffs.json", "r"))[p["simtype"]]
        xy = np.append(np.array(points_low)[np.newaxis, :], np.array(points_high)[np.newaxis, :], axis=0)
        predictions.append(powerlaw2d(xy, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], coeffs[5], coeffs[6]))
    return predictions
    


def predict_with_nn(p):
    """Returns neural network predictions on test set"""
    ### load nn model
    nn_model = torch.load(f"{p['model_path']}/{p_to_filename(p)}.pt", map_location=torch.device("cpu"))

    ### load normalization factors which were used during model training
    _, _, _, _, _, _, mean_x, mean_y, std_x, std_y = load_nn_dataset(p)
    
    ### load data normalized by all images (so incorrect normalization)
    p_all = p.copy()
    p_all["simtype"] = "all"
    testx, _, _, _, _, _, wrong_mean_x, _, wrong_std_x, _ = load_nn_dataset(p_all)

    predictions = []
    for i in range(len(testx)):
        with torch.no_grad():
            #scale back to normal physical values and normalize with mean and std used during training
            testset = (testx[i] * wrong_std_x[np.newaxis, :, np.newaxis, np.newaxis] + wrong_mean_x[np.newaxis, :, np.newaxis, np.newaxis] - mean_x[np.newaxis, :, np.newaxis, np.newaxis]) / std_x[np.newaxis, :, np.newaxis, np.newaxis] 
            ### make predictions
            predictions.append(nn_model(torch.Tensor(testset)).squeeze(1) * std_y + mean_y)
    return predictions




### load existing predictions, if they exist
try:
    with open("/home/tol/Documents/Thesis/data/" + "predictions.json", 'r') as filepath:
        predictions = json.load(filepath)
except:
    predictions = {}

def save_predictions(prediction_dict):
    with open("/home/tol/Documents/Thesis/data/" + "predictions.json", 'w') as filepath:
        json.dump(prediction_dict, filepath, indent=4)


def predict(p):
    """Performs and saves predictions of linear and NN models in dictionary under specific key"""

    ### Calculate prediction
    linear_predictions = predict_with_scaling_relations(p)
    nn_predictions = predict_with_nn(p)
    
    ### add predictions to dictionary
    predictions["linear_predictions"+p_to_filename(p)] = [l.tolist() for l in linear_predictions]
    predictions["nn_predictions"+p_to_filename(p)] = [l.tolist() for l in nn_predictions]

    ### intermediate save
    save_predictions(predictions)


### MAIN LOOP
for noisy in [False, True]:
    p["noisy"] = noisy

    # ### single
    # p["simtype"] = "single"
    # for trainmodel in simulation_variations:
    #     ### set parameters and name
    #     p["model"] = trainmodel
    #     predict(p)
    
    ### all but
    p["simtype"] = "all_but"
    for trainmodel in ["HYDRO_FIDUCIAL"]:
        ### set parameters and name
        p["model"] = trainmodel
        predict(p)
    
    # ### All
    # p["simtype"] = "all"
    # ### set parameters and name
    # p["model"] = "HYDRO_FIDUCIAL"
    # predict(p)


    # ### Extremes
    # p["simtype"] = "extremes"
    # ### set parameters and name
    # p["model"] = "HYDRO_FIDUCIAL"
    # predict(p)