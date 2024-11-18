import importlib

def init(config, *args, **kwargs):
    assert "type" in config 
    assert "args" in config
    return _init(config['type'], config['args'], *args, **kwargs)

def _init(path: str, config: dict, *args, **kwargs):
    p = path.split(".")
    package = ".".join(p[:-1])
    module = p[-1]
    return getattr(importlib.import_module(package), module)(*args, **kwargs, **config)