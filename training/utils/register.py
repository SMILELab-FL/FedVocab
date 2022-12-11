class Registry:
    mapping = {
        "state": {}
    }

    @classmethod
    def register(cls, name, obj):
        cls.mapping["state"][name] = obj

    @classmethod
    def get(cls, name, default=None, no_warning=False):
        original_name = name
        name = name.split(".")
        value = cls.mapping["state"]
        for subname in name:
            value = value.get(subname, default)
            if value is default:
                break

        if (
                "writer" in cls.mapping["state"]
                and value == default
                and no_warning is False
        ):
            cls.mapping["state"]["writer"].warning(
                "Key {} is not present in registry, returning default value "
                "of {}".format(original_name, default)
            )
        return value

    @classmethod
    def unregister(cls, name):
        return cls.mapping["state"].pop(name, None)

    @classmethod
    def get_keys(cls):
        keys = list(cls.mapping["state"].keys())
        return keys

registry = Registry()
