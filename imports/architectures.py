def get_architecture(p):
    if p["channel"] == "2chan":
        in_channels = 2
    elif p["channel"] == "low" or p["channel"] == "high":
        in_channels = 1


    layers = []

    for layer in range(p["conv_layers"]):
        filters = p["base_filters"] * (2**layer)
        layers.append({"type":"conv", "in_channels":in_channels, "out_channels":filters, "kernel_size":p["kernel_size"], "stride":1, "padding": int(p["kernel_size"]/2)})
        layers.append({"type":"leaky_relu", "slope":p["leaky_slope"]})
        if p["use_batch_norm"]:
            layers.append({"type":"batch_norm", "nr_features":filters, "momentum":p["bn_momentum"]})

        for conv in range(p["convs_per_layer"]):
            layers.append({"type":"conv", "in_channels":filters, "out_channels":filters, "kernel_size":p["kernel_size"], "stride":1, "padding": int(p["kernel_size"]/2)})
            layers.append({"type":"leaky_relu", "slope":p["leaky_slope"]})
            if p["use_batch_norm"]:
                layers.append({"type":"batch_norm", "nr_features":filters, "momentum":p["bn_momentum"]})
        
        if layer != (p["conv_layers"]-1):
            layers.append({"type":"pool", "kernel_size":2, "stride":2, "padding": 0})
            in_channels = filters

    layers.append({"type":"flatten"})
    layers.append({"type":"dropout", "p":p["dropout"]})
    filter_size = p["resolution"] * (0.5**(p["conv_layers"]-1))
    nr_features = int(filter_size*filter_size * filters)
    # if 256*nr_features < 2**20: #if an intermediate layer with 256 features does not create more than 1 million free parameters, make that layer
    #     layers.append({"type":"fc", "in_features":int(nr_features), "out_features":256})
    #     layers.append({"type":"leaky_relu", "slope":p["leaky_slope"]})
    #     nr_features = 256
    free_params = 2**22
    if free_params/nr_features > 10: #for an intermediate layer which creates 2**22=4million weights, would its amount of features be more than 10, in that case make a layer with 4 million tunable parameters
        layers.append({"type":"fc", "in_features":nr_features, "out_features":int(free_params/nr_features)})
        layers.append({"type":"leaky_relu", "slope":p["leaky_slope"]})
        if p["use_batch_norm"]:
            layers.append({"type":"batch_norm_1d", "nr_features":int(free_params/nr_features), "momentum":p["bn_momentum"]})
        nr_features = int(free_params/nr_features)
    layers.append({"type":"fc", "in_features":int(nr_features), "out_features":1})





    # layers = [
    #     {"type":"conv", "in_channels":in_channels, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
    #     {"type":"leaky_relu", "slope":p["leaky_slope"]},
    #     {"type":"pool", "kernel_size":3, "stride":2, "padding": 1},
    #     {"type":"conv", "in_channels":20, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
    #     {"type":"leaky_relu", "slope":p["leaky_slope"]},
    #     {"type":"pool", "kernel_size":3, "stride":2, "padding": 1},
    #     {"type":"conv", "in_channels":20, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
    #     {"type":"leaky_relu", "slope":p["leaky_slope"]},
    #     {"type":"pool", "kernel_size":3, "stride":2, "padding": 1},

    #     {"type":"flatten"},
    #     {"type":"fc", "in_features":8*8*20, "out_features":50},
    #     {"type":"leaky_relu", "slope":p["leaky_slope"]},
    #     {"type":"fc", "in_features":50, "out_features":1}
    # ]

    return layers

"""
examples:

{"type":"conv", "in_channels":in_channels, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
{"type":"pool", "kernel_size":3, "stride":2, "padding": 1},
{"type":"fc", "in_features":64*64*20, "out_features":1},
{"type":"batch_norm", "conv_features":20},
{"type":"relu"},
{"type":"leaky_relu", "slope":0.01},
{"type":"flatten"},
{"type":"dropout", "p":0.5},
"""



