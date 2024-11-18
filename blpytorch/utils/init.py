import importlib

def init(config, *args, **kwargs):
    """
    Initialize object using config
    where the config needs to include key `type` and `args`
    if additional information is provided like args and kwargs, it
    will be used to intialize the object
    """
    def _init(path: str, config: dict, *args, **kwargs):
        p = path.split(".")
        package = ".".join(p[:-1])
        module = p[-1]
        return getattr(importlib.import_module(package), module)(*args, **kwargs, **config)
    assert "type" in config 
    assert "args" in config
    return _init(config['type'], config['args'], *args, **kwargs)
