import numpy as np

from utils.enums import AttributeType

def fourier_encoding(x, frequencies):
    sin_part = np.sin(frequencies * x * np.pi)
    cos_part = np.cos(frequencies * x * np.pi)
    return np.concatenate([sin_part, cos_part]).astype(np.float32)

def recalculate_attribute_dimensions(attribute_types, vector_size, sort=False, match_numerical=False):
    if sort:
        sorted_attribute_types = sorted(attribute_types, key=lambda x: x.value)
    else:
        sorted_attribute_types = attribute_types

    attribute_dims = []
    for attribute_type in sorted_attribute_types:
        if attribute_type == AttributeType.CATEGORICAL:
            attribute_dims.append(vector_size)
        else:
            if match_numerical:
                attribute_dims.append(vector_size)
            else:
                attribute_dims.append(1)
    print(f"New Attribute Dimensions: {attribute_dims}")
    return np.array(attribute_dims)