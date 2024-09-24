class SettingsMultiTask():
    def __init__(self, perspective_weights, max_length, attribute_perspectives, attribute_types, attribute_dims) -> None:
        self.perspective_weights = perspective_weights
        self.max_length = max_length
        self.attribute_perspectives = attribute_perspectives
        self.attribute_types = attribute_types
        self.attribute_dims = attribute_dims
