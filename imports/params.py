
p = {
    "base_flamingo_path":"/net/hypernova/data2/FLAMINGO",
    "base_data_path":"/net/dodder/data2/tol/obs_data",#"/home/tol/Documents/Thesis/data",
    "base_model_path":"/home/tol/Documents/Thesis/models",
    "soapfile": "halo_properties_0077.hdf5",
    "simsize":"L1000N1800",
    "cosmology":"HYDRO_FIDUCIAL",
    "selection_type":"SO/500_crit",
    "selection_type_name":"SO_500crit",
    
    "resolution":64,
    "obs_radius":2,#Mpc
    
    "nr_epochs":2,
    "lr":0.00001,
    "batch_size":16
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