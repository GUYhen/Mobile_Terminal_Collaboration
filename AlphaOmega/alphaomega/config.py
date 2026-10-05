import yaml


class Config(dict):
    def __init__(self, data=None):
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = _wrap(value)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key) from None

    def __setattr__(self, key, value):
        self[key] = _wrap(value)


def _wrap(value):
    if isinstance(value, dict) and not isinstance(value, Config):
        return Config(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _merge(node, update):
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(node.get(key), dict):
            _merge(node[key], value)
        else:
            node[key] = _wrap(value)


def _set_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for key in keys[:-1]:
        node = node.setdefault(key, Config())
    node[keys[-1]] = _wrap(value)


def load_config(path, scenario=None, overrides=()):
    with open(path, "r", encoding="utf-8") as f:
        cfg = Config(yaml.safe_load(f))
    if scenario is not None:
        _merge(cfg, cfg.scenarios[scenario])
    for item in overrides:
        key, raw = item.split("=", 1)
        _set_path(cfg, key.strip(), yaml.safe_load(raw))
    for key, value in cfg.datasets[cfg.data.name].items():
        cfg.data.setdefault(key, value)
    return cfg
