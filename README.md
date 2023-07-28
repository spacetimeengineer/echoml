# Download echoml

1.) Download the zip file provided and unzip and open terminal outside directory.


2.) Rename and navigate to 'echoml' directory

    $ mv echoml-master echoml && cd echoml

4.) Install package

    $ pip3 install .


# Example Usage

1.) Navigate to directory with echoml.py script :

    $ python3 echoml kmeans_config.json


    
2.) Modify json config (kmeans_config.json) to modify KMeans Cluster Maps

```json

{
    "path_config" : {
        "EK_data_path" : "/resources",
        "EK_data_filenames" : ["D20090405-T114914.raw"],
        "save_path" : "generated"
        },
    "sonar_model" : "EK60",
    "kmeans_config" : {
        "frequency_list" : [

            "120kHz",
            "200kHz"
        ],
        "cluster_count" : 6,
        "random_state" : 42,
        "model" : "DIRECT",
        "n_init" : 100,
        "max_iter" : 300
    },
    "plotting" : {
        "color_map" : "jet",
        "plot" : true
    },
    "data_reduction" : {
        "data_reduction_type" : "sample_number",
        "range_meter_bin" : 2,
        "ping_time_bin" : "2S",
        "range_sample_num" : 1,
        "ping_num" : 1
    },
    "noise_removal" : true,
    "sub_selection" : {
        "rectangular" : {
            "ping_time_begin" : null,
            "ping_time_end" : null,
            "range_sample_begin" : null,
            "range_sample_end" : null
        },
        "region_files" : [
            "resources/zooplankton_patch.EVR",
            "resources/zooplankton_patch1.EVR",
            "resources/zooplankton_patch2.EVR"
        ],
        "line_files" : [
            "resources/zooplankton_patch0.EVL",
            "resources/zooplankton_patch2.EVL"
        ]
    }
}


```



