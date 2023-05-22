# Download echoml

1.) Download the zip file provided and then unzip and navigate to folder.


2.) Rename and navigate to 'echoml' directory

    $ mv echoml-master echoml && cd echoml

4.) Install package

    $ pip3 install .


# Example (app.py provided)

```python

import echoml as eml


eml.KMClusterMap(
    filepath = "./resources/D20090405-T114914.raw",
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


```



