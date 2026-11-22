
class Params:
    _instance = None
    _params = {}

    def __new__(cls):
        if not cls._instance:
            cls._instance = super(Params, cls).__new__(cls)
        return cls._instance

    def __getitem__(self, key):
        return self._params.get(key)

    def __setitem__(self, key, value):
        self._params[key] = value

    def get_params(self):
        return self._params

    def update_params(self, new_params):
        self._params.update(new_params)

    def keys(self):
        return self._params.keys()

    def values(self):
        return self._params.values()

    def items(self):
        return self._params.items()

    @classmethod
    def reset_instance(cls):
        cls._instance = None
