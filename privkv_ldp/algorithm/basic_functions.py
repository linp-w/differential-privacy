import numpy as np
import random


def generate_binary_random(pr, first, second):
    if np.random.random() <= pr:
        return first
    else:
        return second


def mutil_prob_perturbation(candidate, p):
    """Multi-probability perturbation function
    Select different elements with different probabilities, such as 1 with probability 0.2, -1 with probability 0.3, and 0 with probability 0.5
    ([1,-1,0],[0.2,0.3,0.5])

    Args:
        candidate (list): elements
        p (list): probabilities

    Returns:
        value
    """
    return np.random.choice(candidate, p=p)


def correct(low, high, value):
    """correct the value in range [low, high]

    Args:
        low (_type_): _description_
        high (_type_): _description_
        value (_type_): _description_
    """
    if value < low:
        value = low
    elif value > high:
        value = high
    return value


def value_perturbation_primitive(value, epsilon):
    """Algorithm 2 in the paper, perturb the value of a KV pair

    Args:
        value (_type_): _description_
        epsilon (_type_): privacy budget

    Returns:
        the perturbed value
    """
    p_discretization = (1+value)/2
    p_perturbation = (np.e**epsilon)/(np.e**epsilon+1)
    v_star_d = generate_binary_random(p_discretization, 1, -1)
    v_star_p = generate_binary_random(p_perturbation, v_star_d, -1*v_star_d)
    return v_star_p


def local_perturbation_protocol(k, v, d, epsilon_k, epsilon_v):
    """local perturbation protocol for a user

    Args:
        k (_type_): user's key set
        v (_type_): user's value set
        d (_type_): the whole key set dimension
        epsilon_k (_type_): privacy budget of key
        epsilon_v (_type_): privacy budget of value

    Returns:
        the perturbed KV pair <kj, v> of the j-th key and index j
    """
    j = random.randint(1, d)
    p_perturb = (np.e**epsilon_k)/(np.e**epsilon_k+1)

    if j in k:
        index = k.index(j)
        v_star = value_perturbation_primitive(v[index], epsilon_v)
        if np.random.random() < p_perturb:
            k_p = 1
            v_p = v_star
        else:
            k_p = 0
            v_p = 0
    else:
        m = np.random.random()*2-1
        v_star = value_perturbation_primitive(m, epsilon_v)
        if np.random.random() < p_perturb:
            k_p = 0
            v_p = 0
        else:
            k_p = 1
            v_p = v_star
    return (k_p, v_p, j)

