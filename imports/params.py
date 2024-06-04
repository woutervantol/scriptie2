
p = {
    "flamingo_path":"/net/hypernova/data2/FLAMINGO",

    # "data_path":"/net/dodder/data2/tol/obs_data/",
    # "model_path":"/home/tol/Documents/Thesis/models/",
    # "old_model_path":"/home/tol/Documents/Thesis/models_old/",
    # "ray_log_path":"/home/tol/Documents/Thesis/tune_log",
    "data_path":"/home/s2041340/data1/data/",
    "model_path":"/home/s2041340/data1/models/",
    "ray_log_path":"/home/s2041340/data1/tune_log",

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
    "modules":7.0,
    
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
    # "conv_channels":16,
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




# {'type': 'conv', 'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'skip_connection': True},
# {'type': 'relu'},
# {'type': 'pool', 'kernel_size': 2, 'stride': 2, 'padding': 0},
# {'type': 'conv', 'in_channels': 64, 'out_channels': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'skip_connection': True},
# {'type': 'relu'},
# {'type': 'pool', 'kernel_size': 2, 'stride': 2, 'padding': 0},
# {'type': 'fc', 'in_features': 128 * 8 * 8, 'out_features': 256},
# {'type': 'relu'},
# {'type': 'fc', 'in_features': 256, 'out_features': 10},
# {'type': 'softmax'}