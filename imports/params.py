### Default parameters
p = {
    "flamingo_path":"/net/hypernova/data2/FLAMINGO",

    ### DODDER paths:
    "data_path":"/net/dodder/data2/tol/obs_data/",
    "model_path":"/home/tol/Documents/Thesis/models/",
    "old_model_path":"/home/tol/Documents/Thesis/models_old/",
    "ray_log_path":"/home/tol/Documents/Thesis/tune_log",
    ### ALICE paths:
    # "data_path":"/home/s2041340/data1/data/",
    # "model_path":"/home/s2041340/data1/models/",
    # "ray_log_path":"/home/s2041340/data1/tune_log",

    "soapfile": "halo_properties_0077.hdf5",
    "snapshot":"flamingo_0077/flamingo_0077.hdf5",
    "snapshot_folder": "snapshots",
    "simsize":"L1000N1800",
    "model":"HYDRO_FIDUCIAL",
    "selection_type":"SO/500_crit",
    "selection_type_name":"SO_500crit",

    "simtype":"single",
    "noisy":False,
    
    "resolution":64,
    "obs_radius":2,#Mpc
    "nr_uniform_bins_obs_data":20,
    "redshift":0.15,
    "diameter":0.36, #m
    "obs_time":100000.0, #s
    "modules":7,
    
    "test_size":0.1,
    "val_size":0.2,
    "nr_epochs":2,
    "lr":0.00001,
    "lrfactor":0.2,
    "lrpatience":10,
    "L2":0.01,
    "batch_size":16,
    "bn_momentum":0.1,

    "channel":"2chan",
    "kernel_size":3, #must be uneven
    "base_filters":16,
    "leaky_slope":0.01,
    "dropout":0.1,
    "conv_layers":2,
    "convs_per_layer":2,
    "use_pooling":False,
    "use_batch_norm":False,

    "search_alg": "BOHB",
    "time_budget": 60*5, #in seconds
}