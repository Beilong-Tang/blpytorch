import yaml

def get_env(config_path:str):
    """
    config_path: str to yaml config
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(config_path)
    config = AttrDict(**config)
    return config



class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __getattribute__(self, name: str):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return None

if __name__ == "__main__":
    config = yaml.safe_load("a: 2")
    config = AttrDict(**config)
    print(config.a)
    print(config.get("a"))
