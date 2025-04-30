import yaml
import os
import argparse
from pathlib import Path
from datetime import datetime


def load_yaml(file_path):
    """
    Load a YAML file and return its contents as a dictionary.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        dict: Parsed YAML content.
    """
    expanded_path = os.path.expandvars(file_path)  # Expand environment variables like $HOME
    with open(expanded_path, "r") as file:
        return yaml.safe_load(file)


def save_yaml(data, output_path):
    """
    Save a dictionary as a YAML file.

    Args:
        data (dict): Data to save.
        output_path (str): Path to save the YAML file.
    """
    # Ensure the kmap_exports directory exists
    default_export_dir = os.path.expanduser("~/Documents/kmap_exports")
    os.makedirs(default_export_dir, exist_ok=True)

    # Ensure the directory for the output path exists
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w") as  file:
        yaml.dump(data, file, default_flow_style=False)
    print(f"Updated YAML saved to: {output_path}")


def parse_metadata(yaml_data):
    """
    Parse the metadata from the YAML file and return a structured dictionary.

    Args:
        yaml_data (dict): Parsed YAML content.

    Returns:
        dict: Structured metadata.
    """
    metadata = {
        "path_configuration": yaml_data.get("path_configuration", {}),
        "kmeans_configuration": yaml_data.get("kmeans_configuration", {}),
        "pre_clustering_model": yaml_data.get("pre_clustering_model", {}),
        "plotting": yaml_data.get("plotting", {}),
        "mvbs_data_reduction": yaml_data.get("mvbs_data_reduction", {}),
        "noise_removal": yaml_data.get("noise_removal", {}),
        "subset_selection": yaml_data.get("subset_selection", {}),
        "dependencies": yaml_data.get("dependencies", {}),
    }
    return metadata


def main():
    # Command-line argument parser
    parser = argparse.ArgumentParser(description="Parse and override a YAML metadata file.")
    
    parser.add_argument(
        "--config_path",
        required=False,
        help="Path to the YAML file containing metadata and arguments.",
    )
    parser.add_argument("--raw_path", help="Override the file path.")
    parser.add_argument("--save_path", help="Override the save path.")
    parser.add_argument("--frequency_list", nargs="+", help="Override the frequency list.")
    parser.add_argument("--cluster_count", type=int, help="Override the cluster count.")
    parser.add_argument("--random_state", type=int, help="Override the random state.")
    parser.add_argument("--model", help="Override the pre-clustering model.")
    parser.add_argument("--color_map", help="Override the colormap.")
    parser.add_argument("--plot", action="store_true", help="Override the plot flag.")
    parser.add_argument("--plot_relevent_echograms", action="store_true", help="Override the plot relevant echograms flag.")
    parser.add_argument("--data_reduction_type", help="Override the data reduction type.")
    parser.add_argument("--range_meter_bin", type=int, help="Override the range meter bin.")
    parser.add_argument("--ping_time_bin", help="Override the ping time bin.")
    parser.add_argument("--range_sample_num", type=int, help="Override the range sample number.")
    parser.add_argument("--ping_num", type=int, help="Override the ping number.")
    parser.add_argument("--remove_noise", action="store_true", help="Override the noise removal flag.")
    parser.add_argument("--ping_time_begin", type=int, help="Override the ping time begin.")
    parser.add_argument("--ping_time_end", type=int, help="Override the ping time end.")
    parser.add_argument("--range_sample_begin", type=int, help="Override the range sample begin.")
    parser.add_argument("--range_sample_end", type=int, help="Override the range sample end.")
    args = parser.parse_args()

    if not args.config_path and not args.raw_path:
        exit("Error: Either --config_path or --raw_path must be provided.")
    if not args.config_path and args.raw_path:
        # Create a default metadata structure
        metadata = {
            "path_configuration": {
                "raw_path": args.raw_path,
                "save_path": os.path.expanduser("~/Documents/kmap_exports"),
            },
            "kmeans_configuration": {
                "frequency_list": [],
                "cluster_count": 5,
                "random_state": 42,
            },
            "pre_clustering_model": {
                "model": None,
            },
            "plotting": {
                "color_map": "viridis",
                "plot": False,
                "plot_relevent_echograms": False,
            },
            "mvbs_data_reduction": {
                "data_reduction_type": None,
                "range_meter_bin": 0,
                "ping_time_bin": None,
                "range_sample_num": 0,
                "ping_num": 0,
            },
            "noise_removal": {
                "remove_noise": False,
            },
            "subset_selection": {
                "ping_time_begin": None,
                "ping_time_end": None,
                "range_sample_begin": None,
                "range_sample_end": None,
            },
            "dependencies": {},
        }

        # Save the default metadata to a YAML file
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_filename = f"metadata_{timestamp}.yaml"
        output_path = Path(metadata["path_configuration"]["save_path"]) / output_filename
        save_yaml(metadata, output_path)
        
    if args.config_path and not args.raw_path:

        # Load YAML file
        yaml_data = load_yaml(args.config_path)

        # Parse metadata
        metadata = parse_metadata(yaml_data)

        # Apply overrides
        if args.raw_path:
            metadata["path_configuration"]["raw_path"] = args.raw_path
            args.config_path 
        if args.save_path:
            metadata["path_configuration"]["save_path"] = args.save_path
        if args.frequency_list:
            metadata["kmeans_configuration"]["frequency_list"] = args.frequency_list
        if args.cluster_count:
            metadata["kmeans_configuration"]["cluster_count"] = args.cluster_count
        if args.random_state:
            metadata["kmeans_configuration"]["random_state"] = args.random_state
        if args.model:
            metadata["pre_clustering_model"]["model"] = args.model
        if args.color_map:
            metadata["plotting"]["color_map"] = args.color_map
        if args.plot:
            metadata["plotting"]["plot"] = args.plot
        if args.plot_relevent_echograms:
            metadata["plotting"]["plot_relevent_echograms"] = args.plot_relevent_echograms
        if args.data_reduction_type:
            metadata["mvbs_data_reduction"]["data_reduction_type"] = args.data_reduction_type
        if args.range_meter_bin:
            metadata["mvbs_data_reduction"]["range_meter_bin"] = args.range_meter_bin
        if args.ping_time_bin:
            metadata["mvbs_data_reduction"]["ping_time_bin"] = args.ping_time_bin
        if args.range_sample_num:
            metadata["mvbs_data_reduction"]["range_sample_num"] = args.range_sample_num
        if args.ping_num:
            metadata["mvbs_data_reduction"]["ping_num"] = args.ping_num
        if args.remove_noise:
            metadata["noise_removal"]["remove_noise"] = args.remove_noise
        if args.ping_time_begin:
            metadata["subset_selection"]["ping_time_begin"] = args.ping_time_begin
        if args.ping_time_end:
            metadata["subset_selection"]["ping_time_end"] = args.ping_time_end
        if args.range_sample_begin:
            metadata["subset_selection"]["range_sample_begin"] = args.range_sample_begin
        if args.range_sample_end:
            metadata["subset_selection"]["range_sample_end"] = args.range_sample_end

    # Generate a new filename with a timestamp
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_filename = f"updated_metadata_{timestamp}.yaml"
    output_path = Path(metadata["path_configuration"].get("save_path", ".")) / output_filename

    # Save the updated YAML
    save_yaml(metadata, output_path)


if __name__ == "__main__":
    main()
    

# Behavior : Either a raw file path or a config path (yaml config) is required for the script to run. If only a raw file is proovided, defaults will be assumed and the script will run and create a new config file in the save path. It will perfgomr an override if it is in the same path and the same. Otherwise a new file name cinfig will be named with values encoded in the file name.