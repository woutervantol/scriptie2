from imports.data import *
from imports.params import p
from scipy.optimize import curve_fit


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
    data = Data(p)
    filename =  p_to_filename(p)
    luminosity = np.empty((0, 2))
    mass = np.empty((0))
    if p["simtype"] != "single":
        if p["simtype"] != "extremes":
            for model in ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_JETS_published", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
                if p["simtype"] == "all_but" and p["model"] == model:
                    continue
                p_temp = p.copy()
                p_temp["model"] = model
                data = Data(p_temp)
                filename = p_to_filename(p_temp)
                data.load_dataset(filename)
                luminosity = np.append(luminosity, data.soap_file[f"SO/500_crit/XRayPhotonLuminosityWithoutRecentAGNHeating"][:][data.indices, :2], axis=0)
                mass = np.append(mass, data.masses)
        else:
            for model in ["HYDRO_WEAK_AGN", "HYDRO_STRONGEST_AGN"]:
                p_temp = p.copy()
                p_temp["model"] = model
                data = Data(p_temp)
                filename = p_to_filename(p_temp)
                data.load_dataset(filename)
                luminosity = np.append(luminosity, data.soap_file[f"SO/500_crit/XRayPhotonLuminosityWithoutRecentAGNHeating"][:][data.indices, :2], axis=0)
                mass = np.append(mass, data.masses)
    else:
        data.load_dataset(filename)
        luminosity = data.soap_file[f"SO/500_crit/XRayPhotonLuminosityWithoutRecentAGNHeating"][:][data.indices]
        mass = data.masses

    print("data loaded")
    flux_ratio, fov = get_flux_ratio(p)
    mass = np.log10(mass)
    luminosity = np.log10(luminosity*flux_ratio*p["obs_time"])


    coeff_low, var = curve_fit(power_law, luminosity[:,0], mass, sigma=0.1, maxfev=300000, p0=[5, 13.5, 0.3, 0.7])
    # print("L0, M0, p1, p2 for low energy band: ", coeff_low)
    coeff_high, var = curve_fit(power_law, luminosity[:,1], mass, sigma=0.1, maxfev=300000, p0=[5, 13.5, 0.3, 0.7])
    # print("L0, M0, p1, p2 for high energy band: ", coeff_high)


    datapoints2d = np.append(luminosity[:,0][np.newaxis, :], luminosity[:,1][np.newaxis, :], axis=0)
    coeff2d, var = curve_fit(powerlaw2d, datapoints2d, mass, sigma=0.1, p0=[coeff_low[0], coeff_low[2], coeff_low[3], coeff_high[0], coeff_high[2], coeff_high[3], 14])
    print(coeff2d)
    return coeff2d

    # coeffdict = {}
    # coeffdict["low"] = {}
    # coeffdict["low"]["L0"] = coeff2d[0]
    # coeffdict["low"]["M0"] = coeff2d[1]
    # coeffdict["low"]["p1"] = coeff2d[2]
    # coeffdict["low"]["p2"] = coeff2d[3]
    # coeffdict["high"] = {}
    # coeffdict["high"]["L0"] = coeff2d[4]
    # coeffdict["high"]["M0"] = coeff2d[5]
    # coeffdict["high"]["p1"] = coeff2d[6]
    # coeffdict["high"]["p2"] = coeff2d[7]
    # coeffdict["2d"] = {}
    # coeffdict["2d"]["A1"] = coeff2d[8]
    # coeffdict["2d"]["A2"] = coeff2d[9]



p["redshift"] = 0.15
p["simtype"] = "all_but"
p["model"] = "HYDRO_STRONG_JETS_published"
print(get_coeffs(p))
coeffdict = {}
coeffdict["single"] = {}
# print(get_coeffs(p))

p["simtype"]="single"
print("Single:")
for model in ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
    print("Begin", model)
    p["model"] = model
    coeffdict["single"][model] = get_coeffs(p)

p["simtype"]="all_but"
print("All but:")
for model in ["HYDRO_FIDUCIAL", "HYDRO_JETS_published", "HYDRO_STRONG_AGN", "HYDRO_STRONG_SUPERNOVA", "HYDRO_STRONGER_AGN", "HYDRO_STRONGER_AGN_STRONG_SUPERNOVA", "HYDRO_STRONGEST_AGN", "HYDRO_WEAK_AGN"]:
    print("Begin", model)
    p["model"] = model
    coeffdict["single"][model] = get_coeffs(p)
print(coeffdict)

# import json
# with open(p['model_path'] + "powerlaw_coeffs_photons.json", 'w') as filepath:
#     json.dump(all_coeffs, filepath, indent=4)