import echoml as eml



eml.KMClusterMap(
    filepath = "./D20090405-T114914.raw",
    frequency_list = ['38kHz','70kHz','120kHz','18kHz','200kHz'],
    cluster_count = 24,
    save_path = "km_cluster_maps",
    random_state = 42,
    color_map = "jet",
    range_meter_bin = 2,
    ping_time_bin = '4S',
    range_bin_num = 300,      # Ineffectual
    ping_num = 2,             # Ineffectual
    model = "direct",
    plot = True,
    remove_noise = True       # Ineffectual
    )
