def get_architecture(p):
    """Returns list of dictionaries, with each dictionary a layer containing the layer name and parameter values."""
    
    ### alter input shape for single band model
    if p["channel"] == "2chan":
        in_channels = 2
    elif p["channel"] == "low" or p["channel"] == "high":
        in_channels = 1


    layers = []

    for layer in range(p["conv_layers"]):
        ### double the number of filters for each "set", which is confusingly called a layer in the variable names
        filters = p["base_filters"] * (2**layer)
        layers.append({"type":"conv", "in_channels":in_channels, "out_channels":filters, "kernel_size":p["kernel_size"], "stride":1, "padding": int(p["kernel_size"]/2)})
        layers.append({"type":"leaky_relu", "slope":p["leaky_slope"]})
        if p["use_batch_norm"]:
            layers.append({"type":"batch_norm", "nr_features":filters, "momentum":p["bn_momentum"]})

        ### add a number of convolutional layers in each set
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

    ### add fully connected layer with 2**22 free parameters
    filter_size = p["resolution"] * (0.5**(p["conv_layers"]-1))
    nr_features = int(filter_size*filter_size * filters)
    free_params = 2**22
    ### if the fully connected layer would contain less than 10 features, skip it and go to the output layer immediately
    if free_params/nr_features > 10: 
        layers.append({"type":"fc", "in_features":nr_features, "out_features":int(free_params/nr_features)})
        layers.append({"type":"leaky_relu", "slope":p["leaky_slope"]})
        if p["use_batch_norm"]:
            layers.append({"type":"batch_norm_1d", "nr_features":int(free_params/nr_features), "momentum":p["bn_momentum"]})
        nr_features = int(free_params/nr_features)
    ### output layer
    layers.append({"type":"fc", "in_features":int(nr_features), "out_features":1})


    return layers
