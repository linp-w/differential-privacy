from scipy.special import comb
import numpy as np
import itertools
import math
import random
import matplotlib.pyplot as plt


def generate_binary_random(pr, first, second):
    if random.random() <= pr:
        return first
    else:
        return second


def duchi_method(tp, epsilon):
    """This is the implement of Duchi local differential privacy.

    Args:
        tp (array): refers to a row of the dataset representing the data of a user
        epsilon (float): privacy budget

    Returns:
        _type_: the perturbed tuple of the data of a user

    Notes: there exists an error in the original Duchi's ldp when the dimension of tp is even \
        which does not satisfy epsilon-differential privacy.
    """
    d = len(tp)
    if d % 2 != 0:  # if d is odd
        C_d = 2 ** (d - 1)
        B = (2 ** d + C_d * (math.exp(epsilon) - 1)) / \
            (comb(d - 1, int((d - 1) / 2)) * (math.exp(epsilon) - 1))
    else:           # otherwise
        C_d = 2 ** (d - 1) - comb(d, int(d / 2))
        B = (2 ** d + C_d * (math.exp(epsilon) - 1)) / \
            (comb(d - 1, int(d / 2)) * (math.exp(epsilon) - 1))

    neg_B = (-1) * B
    v = [generate_binary_random(0.5 + 0.5 * tp[j], 1, -1)
         for j in range(d)]      # get the v

    # T+ the set of all tuples (neg_B, B)*d such that (t+) * v > 0
    t_pos = []
    # T- the set of all tuples (neg_B, B)*d such that (t+) * v < 0
    t_neg = []
    for t_star in itertools.product([neg_B, B], repeat=d):
        if np.dot(np.array(t_star), np.array(v)) > 0:
            t_pos.append(t_star)
        else:
            t_neg.append(t_star)

    # sample a Bernoulli variable u that equal 1 with (exp(epsilon)/(exp(epsilon)+1) probability
    # then if u == 1 return T+, otherwise return T-
    if generate_binary_random(math.exp(epsilon) / (math.exp(epsilon) + 1), 1, 0) == 1:
        return random.choice(t_pos)
    else:
        return random.choice(t_neg)


def duchi_method_modified_u(tp, epsilon):
    """This is the implement of the modified Duchi local differential privacy \
    which the last random variable u is (exp(epsilon)*C_d)/((exp(epsilon-)-1)*C_d+2**d) for \
        satisfying epsilon-differential privacy.

    Args:
        tp (array): refers to a row of the dataset representing the data of a user
        epsilon (float): privacy budget

    Returns:
        _type_: the perturbed tuple of the data of a user

    Notes: there exists an error in the original Duchi's ldp when the dimension of tp is even
    """
    d = len(tp)
    if d % 2 != 0:
        C_d = 2 ** (d - 1)
        B = (2 ** d + C_d * (math.exp(epsilon) - 1)) / \
            (comb(d - 1, int((d - 1) / 2)) * (math.exp(epsilon) - 1))
    else:
        C_d = 2 ** (d - 1) - comb(d, int(d / 2))
        B = (2 ** d + C_d * (math.exp(epsilon) - 1)) / \
            (comb(d - 1, int(d / 2)) * (math.exp(epsilon) - 1))

    neg_B = (-1) * B
    v = [generate_binary_random(0.5 + 0.5 * tp[j], 1, -1)
         for j in range(d)]

    t_pos = []
    t_neg = []
    for t_star in itertools.product([neg_B, B], repeat=d):
        if np.dot(np.array(t_star), np.array(v)) > 0:
            t_pos.append(t_star)
        else:
            t_neg.append(t_star)

    if generate_binary_random((math.exp(epsilon) * C_d) / ((math.exp(epsilon) - 1) * C_d + 2 ** d), 1, 0) == 1:
        return random.choice(t_pos)
    else:
        return random.choice(t_neg)


if __name__ == '__main__':
    # set the attribute dimension
    dimension = 10
    # set the number of users or data
    num = 5000
    # set the privacy budget
    epsilon = 0.7
    random.seed(10)

    print('---------------------- sample --------------------------', '\n')
    sample_t = np.array([[random.uniform(-1, 1)
                        for di in range(dimension)] for n in range(num)])
    sample_mean = np.mean(sample_t, axis=0)
    print('the real sample mean is', sample_mean, '\n')

    print('------------------ original duchi ----------------------', '\n')
    duchi_method_t = np.array([duchi_method(tp, epsilon) for tp in sample_t])
    duchi_method_mean = np.mean(duchi_method_t, axis=0)
    print('mean using Duchi\'s method:', duchi_method_mean, '\n')
    # print("absolute error of Duchi's method:",
    #       np.fabs(sample_mean - duchi_method_mean), '\n')
    # print("relative error of Duchi's method:", np.fabs(
    #     np.true_divide(sample_mean - duchi_method_mean, sample_mean)), '\n')

    # print('max expect', np.max(np.fabs(sample_mean-duchi_method_mean)))
    # print('upper O', np.true_divide(
    #     np.sqrt(dimension*np.log(dimension)), (epsilon*np.sqrt(num))))

    print('------------------ modified duchi ----------------------', '\n')
    duchi_method_u_t = np.array(
        [duchi_method_modified_u(tp, epsilon) for tp in sample_t])
    duchi_method_u_mean = np.mean(duchi_method_u_t, axis=0)
    print('mean using modified Duchi\'s method:', duchi_method_u_mean, '\n')

    print('------ compare the absolute error of original duchi with modified duchi ------', '\n')
    plt.plot(np.fabs(sample_mean - duchi_method_mean),
             marker="o", label="duchi")
    plt.plot(np.fabs(sample_mean - duchi_method_u_mean),
             marker="o", color="red", label="duchi_u")
    plt.title("Absolute Error")
    plt.legend()
    plt.show()
