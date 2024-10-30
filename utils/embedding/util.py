import numpy as np

def fourier_encoding(x, frequencies):
    sin_part = np.sin(frequencies * x * np.pi)
    cos_part = np.cos(frequencies * x * np.pi)
    return np.concatenate([sin_part, cos_part]).astype(np.float32)