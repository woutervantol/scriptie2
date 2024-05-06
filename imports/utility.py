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