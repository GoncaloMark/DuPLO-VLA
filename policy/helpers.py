class MetaWrapper:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, key):
        return self.data[key]
    def __repr__(self):
        return f"Wrapper({self.data})"

def prepare_meta(data):
    if isinstance(data, dict):
        if 'shape' in data:
            return MetaWrapper(data)
        return {k: prepare_meta(v) for k, v in data.items()}
    return data

class ConfigWrapper(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)