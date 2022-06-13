from basic_functions import *
import numpy as np


def PrivKV(k, v, d, epsilon_k, epsilon_v):
    """_summary_

    Args:
        k (_type_): all users' sets of key of KV pairs
        v (_type_): all users's sets of value of KV paires
        d (_type_): the dimension of the set of keys 
        epsilon_k (_type_): _description_
        epsilon_v (_type_): _description_
    """
    n = len(k)
    all_kvp = [local_perturbation_protocol(
        k[i], v[i], d, epsilon_k, epsilon_v)for i in range(n)]

    p1 = (np.e**epsilon_k)/(np.e**epsilon_k+1)
    p2 = (np.e**epsilon_v)/(np.e**epsilon_v+1)

    pos = [0 for i in range(d)]
    neg = [0 for i in range(d)]
    count = [0 for i in range(d)]
    for kv in all_kvp:
        if kv[0] == 0:
            continue
        j = kv[2]
        if kv[1] == 1:
            pos[j-1] += 1
            count[j-1] += 1
        if kv[1] == -1:
            neg[j-1] += 1
            count[j-1] += 1

    frequency = np.array(count)/n
    f_k = ((p1-1+frequency)/(2*p1-1)).tolist()

    pos = np.array(pos)
    neg = np.array(neg)
    N = pos + neg
    n1 = N * (p2 - 1) / (2 * p2 - 1) + pos / (2 * p2 - 1)
    n2 = N * (p2 - 1) / (2 * p2 - 1) + neg / (2 * p2 - 1)

    for i in range(d):
        n1[i] = correct(0, N[i], n1[i])
        n2[i] = correct(0, N[i], n2[i])
        if(N[i] == 0):
            N[i] = 1

    m_k = ((n1 - n2) / N).tolist()

    return f_k, m_k
