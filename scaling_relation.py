from imports.data import *
from imports.params import p
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


def power_law(L, L0, M0, p1, p2):
    return np.append(p1*(L[L<L0]-L0)+M0, p2*(L[L>=L0]-L0)+M0)

def powerlaw2d(xy, L0_low, p1_low, p2_low, L0_high, p1_high, p2_high, M0):
    x, y = xy
    fit_low = np.zeros(len(x))
    fit_low[x < L0_low] = p1_low * (x[x < L0_low] - L0_low)
    fit_low[x >= L0_low] = p2_low * (x[x >= L0_low] - L0_low)
    fit_high = np.zeros(len(y))
    fit_high[y < L0_high] = p1_high * (y[y < L0_high] - L0_high)
    fit_high[y >= L0_high] = p2_high * (y[y >= L0_high] - L0_high)
    return fit_low + fit_high + M0

def get_coeffs(p):
    images, masses, radii = load_nn_dataset(p, normalise=False)
    pixel_size = 4 * 1000 / 64  #4Mpc * 1000kpc/Mpc / 64pixels
    lums_low = []
    lums_high = []
    for i in range(len(images)):
        radius = radii[i] / pixel_size
        X, Y = np.ogrid[:p['resolution'], :p['resolution']]
        dist_from_center = np.sqrt((X + 0.5-int(p['resolution']/2))**2+(Y+0.5-int(p['resolution']/2))**2)
        mask_circ = dist_from_center<=radius
        lums_low.append(np.log10(np.sum(10**images[i, 0][mask_circ])))
        lums_high.append(np.log10(np.sum(10**images[i, 1][mask_circ])))

    # masses = np.log10(masses)
    lums_low = np.array(lums_low)
    lums_high = np.array(lums_high)

    coeff_low, var = curve_fit(power_law, lums_low, masses, sigma=0.1, maxfev=300000, p0=[5, 13.5, 0.3, 0.7])
    print("L0, M0, p1, p2 for low energy band: ", coeff_low)
    coeff_high, var = curve_fit(power_law, lums_high, masses, sigma=0.1, maxfev=300000, p0=[5, 13.5, 0.3, 0.7])
    print("L0, M0, p1, p2 for high energy band: ", coeff_high)

    datapoints2d = np.append(lums_low[np.newaxis, :], lums_high[np.newaxis, :], axis=0)
    coeff2d, var = curve_fit(powerlaw2d, datapoints2d, masses, sigma=0.1, p0=[coeff_low[0], coeff_low[2], coeff_low[3], coeff_high[0], coeff_high[2], coeff_high[3], 14])
    print(coeff2d)
    return coeff2d

# def get_coeffs(p):
#     data = Data(p)
#     filename =  p_to_filename(p)
#     luminosity = np.empty((0, 2))
#     mass = np.empty((0))
#     if p["simtype"] != "single":
#         if p["simtype"] != "extremes":
#             for model in ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_JETS_published", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
#                 if p["simtype"] == "all_but" and p["model"] == model:
#                     continue
#                 p_temp = p.copy()
#                 p_temp["model"] = model
#                 data = Data(p_temp)
#                 filename = p_to_filename(p_temp)
#                 data.load_dataset(filename)
#                 luminosity = np.append(luminosity, data.soap_file[f"SO/500_crit/XRayPhotonLuminosityWithoutRecentAGNHeating"][:][data.indices, :2], axis=0)
#                 mass = np.append(mass, data.masses)
#         else:
#             for model in ["HYDRO_WEAK_AGN", "HYDRO_STRONGEST_AGN"]:
#                 p_temp = p.copy()
#                 p_temp["model"] = model
#                 data = Data(p_temp)
#                 filename = p_to_filename(p_temp)
#                 data.load_dataset(filename)
#                 luminosity = np.append(luminosity, data.soap_file[f"SO/500_crit/XRayPhotonLuminosityWithoutRecentAGNHeating"][:][data.indices, :2], axis=0)
#                 mass = np.append(mass, data.masses)
#     else:
#         data.load_dataset(filename)
#         luminosity = data.soap_file[f"SO/500_crit/XRayPhotonLuminosityWithoutRecentAGNHeating"][:][data.indices]
#         mass = data.masses
#     # print("data loaded")
#     flux_ratio, fov = get_flux_ratio(p)
#     mass = np.log10(mass)
#     luminosity = np.log10(luminosity*flux_ratio*p["obs_time"])


#     coeff_low, var = curve_fit(power_law, luminosity[:,0], mass, sigma=0.1, maxfev=300000, p0=[5, 13.5, 0.3, 0.7])
#     # print("L0, M0, p1, p2 for low energy band: ", coeff_low)
#     coeff_high, var = curve_fit(power_law, luminosity[:,1], mass, sigma=0.1, maxfev=300000, p0=[5, 13.5, 0.3, 0.7])
#     # print("L0, M0, p1, p2 for high energy band: ", coeff_high)


#     datapoints2d = np.append(luminosity[:,0][np.newaxis, :], luminosity[:,1][np.newaxis, :], axis=0)
#     coeff2d, var = curve_fit(powerlaw2d, datapoints2d, mass, sigma=0.1, p0=[coeff_low[0], coeff_low[2], coeff_low[3], coeff_high[0], coeff_high[2], coeff_high[3], 14])
#     print(coeff2d)
#     return coeff2d

# import json
# with open(p['model_path'] + "powerlawcoeffs.json", 'r') as filepath:
#     coeffdict = json.load(filepath)

p["redshift"] = 0.15
coeffdict = {}
coeffdict["single"] = {}
coeffdict["all_but"] = {}

p["simtype"]="single"
print("Single:")
for model in ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_JETS_published", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
    print("Begin", model)
    p["model"] = model
    coeffdict["single"][model] = get_coeffs(p).tolist()
    
print(coeffdict)

p["simtype"]="all_but"
print("All but:")
for model in ["HYDRO_FIDUCIAL", "HYDRO_STRONG_AGN"]:
    print("Begin", model)
    p["model"] = model
    coeffdict["all_but"][model] = get_coeffs(p).tolist()
    
print(coeffdict)

p["simtype"] = "all"
p["model"] = model
coeffdict["all"] = get_coeffs(p).tolist()

p["simtype"] = "extremes"
p["model"] = model
coeffdict["extremes"] = get_coeffs(p).tolist()

print(coeffdict)

import json
with open(p['model_path'] + "powerlawcoeffs.json", 'w') as filepath:
    json.dump(coeffdict, filepath, indent=4)