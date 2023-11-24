def get_architecture(network_name, channel="2chan"):
    if channel == "2chan":
        in_channels = 2
    elif channel == "low" or channel == "high":
        in_channels = 1

    archs = {
        "base_conv_network": [
            {"type":"conv", "in_channels":in_channels, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
            {"type":"relu"},
            {"type":"flatten"},
            {"type":"fc", "in_features":64*64*20, "out_features":1}
        ],

        "deep_conv_network": [
            {"type":"conv", "in_channels":in_channels, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
            {"type":"relu"},
            {"type":"conv", "in_channels":20, "out_channels":20, "kernel_size":3, "stride":1, "padding": 1},
            {"type":"relu"},
            {"type":"flatten"},
            {"type":"fc", "in_features":64*64*20, "out_features":1}
        ]
    }
    return archs[network_name]