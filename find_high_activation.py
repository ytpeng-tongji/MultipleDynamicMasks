import numpy as np

def find_high_activation_mask(activation_map, percentile=90):
    threshold = np.percentile(activation_map, percentile)
    mask = np.ones(activation_map.shape)
    mask[activation_map < threshold] = 0

    return mask