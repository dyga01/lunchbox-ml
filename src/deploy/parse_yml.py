"""Parse YAML configuration file."""

import yaml

def parse_yaml_file(yaml_file_path):
    """
    Parses the given YAML file and extracts its sections.

    Args:
        yaml_file_path (str): Path to the YAML configuration file.

    Returns:
        dict: Parsed YAML content as a dictionary.
    """
    with open(yaml_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
