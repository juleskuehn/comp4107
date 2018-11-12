# COMP 4107
# Fall 2018
# Assignment 3
# Jules Kuehn

import numpy as np
import tensorflow as tf
import random
import math
from q2_kmeans import k_means


def rbf_activation(beta, input_vector, center):
    """ 
    beta: a scalar.
    input_vector and center: vectors.
    """
    diff = np.linalg.norm(input_vector - center, ord=1)
    return math.exp(-beta * diff**2)
