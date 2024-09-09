from imports.data import *
from imports.params import p
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def power_law(L, L0, M0, p1, p2):
    """Returns mass prediction for array of luminosities L, given fit parameters L0, M0, p1 and p2"""
    fit = L.copy()
    ### Luminosity below break point uses slope p1, and above the break point p2
    fit[L < L0] = p1*(L[L<L0]-L0)+M0
    fit[L >= L0] = p2*(L[L>=L0]-L0)+M0
    return fit

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

def get_coeffs(p):
    """Returns best fit coefficients for the training set images of the given model variation"""
    images, masses, radii = load_nn_dataset(p, normalise=False)
    pixel_size = 4 * 1000 / 64  #4Mpc * 1000kpc/Mpc / 64pixels
    lums_low = []
    lums_high = []
    ### add pixel values in r_500 for the luminosity values
    for i in range(len(images)):
        radius = radii[i] / pixel_size
        X, Y = np.ogrid[:p['resolution'], :p['resolution']]
        dist_from_center = np.sqrt((X + 0.5-int(p['resolution']/2))**2+(Y+0.5-int(p['resolution']/2))**2)
        mask_circ = dist_from_center<=radius
        lums_low.append(np.log10(np.sum(10**images[i, 0][mask_circ])))
        lums_high.append(np.log10(np.sum(10**images[i, 1][mask_circ])))
    lums_low = np.array(lums_low)
    lums_high = np.array(lums_high)

    ### fit 1D broken power laws for the initial guess of the 2D fit
    bounds = ([0, 10, 0, 0], [10, 20, 2, 2])
    coeff_low, var = curve_fit(power_law, lums_low, masses, sigma=0.1, maxfev=300000, p0=[5, 13.5, 0.3, 0.7], bounds=bounds)
    print("L0, M0, p1, p2 for low energy band: ", coeff_low)
    coeff_high, var = curve_fit(power_law, lums_high, masses, sigma=0.1, maxfev=300000, p0=[5, 13.5, 0.3, 0.7], bounds=bounds)
    print("L0, M0, p1, p2 for high energy band: ", coeff_high)

    ### combine datapoints and perform least squares fit
    bounds = ([0, 0, 0, 0, 0, 0, 10], [10, 2, 2, 10, 2, 2, 20])
    datapoints2d = np.append(lums_low[np.newaxis, :], lums_high[np.newaxis, :], axis=0)
    coeff2d, var = curve_fit(powerlaw2d, datapoints2d, masses, sigma=0.1, p0=[coeff_low[0], coeff_low[2], coeff_low[3], coeff_high[0], coeff_high[2], coeff_high[3], 14], bounds=bounds)
    print(coeff2d)
    return coeff2d

import json
try:
    with open(p['model_path'] + "powerlawcoeffs.json", 'r') as filepath:
        coeffdict = json.load(filepath)
except:
    coeffdict = {}

### create dictionary for best fit parameters
p["redshift"] = 0.15
coeffdict["single"] = {}
coeffdict["all_but"] = {}

### get parameters for single models
p["simtype"]="single"
print("Single:")
for model in simulation_variations:
    print("Begin", model)
    p["model"] = model
    coeffdict["single"][model] = get_coeffs(p).tolist()

### get parameters for all_but models
p["simtype"]="all_but"
print("All but:")
for model in ["HYDRO_FIDUCIAL", "HYDRO_STRONG_AGN"]:
    print("Begin", model)
    p["model"] = model
    coeffdict["all_but"][model] = get_coeffs(p).tolist()

### get parameters for all model
p["simtype"] = "all"
p["model"] = model
coeffdict["all"] = get_coeffs(p).tolist()

### get parameters for extremes model
p["simtype"] = "extremes"
p["model"] = model
coeffdict["extremes"] = get_coeffs(p).tolist()

print(coeffdict)

### save dictionary as json
import json
with open(p['model_path'] + "powerlawcoeffs.json", 'w') as filepath:
    json.dump(coeffdict, filepath, indent=4)


