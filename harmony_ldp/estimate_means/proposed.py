import random
import math
import matplotlib.pyplot as plt
import numpy as np


def generate_binary_random(pr, first, second):
    if random.random() <= pr:
        return first
    else:
        return second


def proposed_method(tp, epsilon):
    """This is the proposed method for handling numeric attributes.

    Args:
        tp (array): refers to a row of the dataset representing the data of a user
        epsilon (float): privacy budget

    Returns:
        array: the perturbed data
    """
    d = len(tp)
    tp_star = [0 for i in range(d)]
    j = random.randint(0, d-1)
    pr = (tp[j] * (math.exp(epsilon) - 1) + math.exp(epsilon) + 1) / \
        (2 * math.exp(epsilon) + 2)
    value = d * (math.exp(epsilon) + 1) / (math.exp(epsilon) - 1)
    if generate_binary_random(pr, 1, 0) == 1:
        tp_star[j] = value
    else:
        tp_star[j] = -1*value
    return tp_star


if __name__ == '__main__':
    # set the attribute dimension
    dimension = 10
    # set the number of users or data
    num = 5000
    # set the privacy budget
    epsilon = 0.7

    print('---------------------- sample --------------------------', '\n')
    sample_t = np.array([[random.uniform(-1, 1)
                        for di in range(dimension)] for n in range(num)])
    sample_mean = np.mean(sample_t, axis=0)
    print('the real sample mean is', sample_mean, '\n')

    print('------------------ proposed method ----------------------', '\n')
    proposed_method_t = np.array(
        [proposed_method(tp, epsilon) for tp in sample_t])
    proposed_method_mean = np.mean(proposed_method_t, axis=0)
    print('mean using proposed method:', proposed_method_mean, '\n')
    print("absolute error of Duchi's method:",
          np.fabs(sample_mean - proposed_method_mean), '\n')
    print("relative error of Duchi's method:", np.fabs(
        np.true_divide(sample_mean - proposed_method_mean, sample_mean)), '\n')
    print('max expect', np.max(np.fabs(sample_mean-proposed_method_mean)))

    print('------ compare the original with proposed ------', '\n')
    plt.plot(sample_mean,
             marker="o", label="sample")
    plt.plot(proposed_method_mean,
             marker="o", color="red", label="proposed")
    plt.title("Mean of the Sample and Proposed")
    plt.legend()
    plt.show()
