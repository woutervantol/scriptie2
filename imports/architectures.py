def get_architecture(p):
    if p["channel"] == "2chan":
        in_channels = 2
    elif p["channel"] == "low" or p["channel"] == "high":
        in_channels = 1

    layers = []

    layers.append({"type":"conv", "in_channels":in_channels, "out_channels":p["conv_channels"], "kernel_size":p["kernel_size"], "stride":1, "padding": int(p["kernel_size"]/2)})
    layers.append({"type":"leaky_relu", "slope":p["leaky_slope"]})
    if p["use_pooling"]:
        layers.append({"type":"pool", "kernel_size":2, "stride":2, "padding": 0})
    if p["use_batch_norm"]:
        layers.append({"type":"batch_norm", "conv_features":p["conv_channels"]})

    for l in range(p["conv_layers"] -1):
        layers.append({"type":"conv", "in_channels":p["conv_channels"], "out_channels":p["conv_channels"], "kernel_size":p["kernel_size"], "stride":1, "padding": int(p["kernel_size"]/2)})
        layers.append({"type":"leaky_relu", "slope":p["leaky_slope"]})
        if p["use_pooling"]:
            layers.append({"type":"pool", "kernel_size":2, "stride":2, "padding": 0})
        if p["use_batch_norm"]:
            layers.append({"type":"batch_norm", "conv_features":p["conv_channels"]})

    layers.append({"type":"flatten"})
    layers.append({"type":"dropout", "p":p["dropout"]})
    nr_features = int(p["resolution"]*p["resolution"] * p["conv_channels"]) if not p["use_pooling"] else int(p["resolution"]*p["resolution"] * p["conv_channels"] / (2**p["conv_layers"])**2)
    layers.append({"type":"fc", "in_features":nr_features, "out_features":1})

    return layers


    # archs = {
    #     "base_conv_network": [
    #         {"type":"conv", "in_channels":in_channels, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
    #         {"type":"relu"},
    #         {"type":"flatten"},
    #         {"type":"fc", "in_features":64*64*20, "out_features":1}
    #     ],

    #     "deep_base_conv_network": [
    #         {"type":"conv", "in_channels":in_channels, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
    #         {"type":"relu"},
    #         {"type":"conv", "in_channels":20, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
    #         {"type":"relu"},
    #         {"type":"conv", "in_channels":20, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
    #         {"type":"relu"},
    #         {"type":"flatten"},
    #         {"type":"fc", "in_features":64*64*20, "out_features":1}
    #     ],

    #     "base_conv_network_leaky": [
    #         {"type":"conv", "in_channels":in_channels, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
    #         {"type":"leaky_relu", "slope":0.05},
    #         {"type":"flatten"},
    #         {"type":"fc", "in_features":64*64*20, "out_features":1}
    #     ],

    #     "param_search": [
    #         {"type":"conv", "in_channels":in_channels, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
    #         {"type":"leaky_relu", "slope":0.05},
    #         {"type":"flatten"},
    #         {"type":"fc", "in_features":64*64*20, "out_features":1}
    #     ],
    # }
    # return archs[network_name]


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



