import numpy as np
from math import pow


def split_genes(genotype):
    return np.split(genotype.genes, 9)


def get_all_weights(genotype) -> np.array:
    weights_binary = split_genes(genotype)
    weights = []
    for weight_binary in weights_binary:
        weights.append(calculate_weight(weight_binary))
    return np.array(weights)


def calculate_weight(weight_binary):
    weight = 0
    for i in np.arange(weight_binary.size - 1, 0, -1):
        try:
            current_elem = (pow(int(weight_binary[i]) * 2, (i - 2)))
            weight += current_elem
        except ValueError:
            weight += 0

    if np.array_equal(weight_binary[0], '1'):
        return weight
    else:
        return -weight