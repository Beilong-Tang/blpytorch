import yaml


def get_value(config_path, value_name):
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)
    return config.get(value_name)
