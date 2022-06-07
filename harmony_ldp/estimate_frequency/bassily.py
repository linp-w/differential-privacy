import math
import numpy as np
import random


def generate_binary_random(pr, first, second):
    if random.random() <= pr:
        return first
    else:
        return second


def bassily_method(tp, k, epsilon, beta):
    """This is the implement of Bassily ans Smith's method of the frequency estimation.

    Args:
        tp (_type_): refers to a column of the the dataset representing a attribute
        k (_type_): refers to the number of possible values for this attribute
        epsilon (_type_): the privacy budget
        beta (_type_): confidence of the error bound

    Returns:
        _type_: Frequency estimate for each of the k values in attribute Aj
    """
    n = len(tp)
    frequency_estimate = []
    gamma = math.sqrt(math.log(2 * k / beta) / (epsilon ** 2 * n))
    # it might be possible that m = 0?
    m = round(math.log(k + 1) * math.log(2 / beta) / (gamma ** 2))
    print("m =", m)
    matrix_value = [-1 / math.sqrt(m), 1 / math.sqrt(m)]
    phi = [[random.choice(matrix_value) for col in range(k)]
           for row in range(m)]

    z_sum = [0 for i_1 in range(m)]
    for i in range(n):
        s = random.randint(0, m - 1)
        c = (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
        if generate_binary_random(math.exp(epsilon) / (math.exp(epsilon) + 1), 1, 0) == 1:
            alpha = c * m * phi[s][tp[i] - 1]
        else:
            alpha = -1 * c * m * phi[s][tp[i] - 1]
        z_sum[s] += alpha

    z_mean = np.array(z_sum) / n
    for l in range(k):
        element = np.dot(np.array(phi)[:, l], z_mean)
        if element >= 0:
            frequency_estimate.append(element)
        else:
            frequency_estimate.append(0)
    print("sum =", sum(frequency_estimate))
    return frequency_estimate


if __name__ == '__main__':
    tp = [1,1,1,1,1,1,1,1,2,2]
    k = 100
    epsilon = 10
    beta = 0.05
    print(bassily_method(tp, k, epsilon, beta))
    print("error bound:", math.sqrt(math.log(k/beta)) / (epsilon * math.sqrt(len(tp))))
