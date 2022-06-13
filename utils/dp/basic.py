import numpy as np
import matplotlib.pyplot as plt


def epsilon_to_probability(epsilon, n=2):
    return np.e ** epsilon / (np.e ** epsilon + n - 1)


def discretization(value, lower, upper):
    """discretiza values

    Args:
        value (_type_): value that needs to be discretized
        lower (_type_): the lower bound of discretized value
        upper (_type_): the upper bound of discretized value

    Raises:
        Exception: _description_

    Returns:
        _type_: the discretized value
    """
    if value > upper or value < lower:
        raise Exception(
            'the range of value is not vaild in Function @Func: discretization')
    p = (value-lower)/(upper-lower)
    rnd = np.random.random()
    return lower if rnd < p else upper


def perturbation(value, perturbed_value, epsilon):
    """perturbation

    Args:
        value (_type_): original value
        perturbed_value (_type_): perturbed value
        epsilon (_type_): privacy budget

    Returns:
        _type_: _description_
    """
    p = epsilon_to_probability(epsilon)
    rnd = np.random.random()
    return value if rnd < p else perturbed_value


def random_response(value, epsilon):
    """random response

    Args:
        value (_type_): _description_
        epsilon (_type_): _description_

    Raises:
        Exception: value is not in [0,1]

    Returns:
        _type_: _description_
    """
    if value not in [0, 1]:
        raise Exception(
            'The input value is not in [0,1] @Func: random_response')
    return perturbation(value, 1-value, epsilon)


def random_response_adjust(sum, N, epsilon):
    """Correct the result of the random response
        sum = x*p + (N-x)(1-p)
    Args:
        sum (_type_): the number of 1s in the received data
        N (_type_): total number of data
        epsilon (_type_): the actual number of 1s

    Returns:
        _type_: _description_
    """
    p = epsilon_to_probability(epsilon)
    return (sum + p*N - N) / (2*p - 1)


# 均值估计思路
# 1. 假如有N个数，每个数分布在[0,1]之间（如果分布在[0,m]，则归一化为[0,1] ‘除以m’）。
# 2. 将每个数离散化到0或1。
# 3. 对离散化的数据做差分隐私处理（Random Response）并发送。
# 4. 服务器收集到DP处理后的数据求和。
# 5. 服务器对求和的数据校正
def mean_estimation_experiment():
    # generated data
    data = np.clip(np.random.normal(loc=0.5, scale=0.2, size=[10000]), 0, 1)
    print("this is generated data\n", data)

    mean = np.average(data)
    print("the mean of original data is: ", mean)

    epsilon = 1

    discretized_data = [discretization(
        value=value, lower=0, upper=1) for value in data]
    dp_data = [random_response(value=value, epsilon=epsilon)
               for value in discretized_data]

    cnt_one = np.sum(dp_data)
    est_one = random_response_adjust(
        sum=cnt_one, N=len(dp_data), epsilon=epsilon)
    est_mean = est_one / len(dp_data)

    print("the estimated mean is: ", est_mean)


# 直方图估计思路
# 假设有m个bucket
# 1. 每个用户用onehot编码自己的数据，即建立一个0向量，并且第i位为1，其中i是自己拥有的数据。
# 2. 对每一位加上 ε/2 的噪声，并发送数据。
# 3. 服务器计算每一位上收到数据的求和，并进行校正，估计出该位上1的个数表示该数据的频数。
def random_response_for_hist(user_vector, epsilon):
    """
    每个用户对自己的数据进行random response操作
    :param user_vector: 
    :param epsilon: privacy budget
    :return: 
    """
    for i in range(len(user_vector)):
        user_vector[i] = random_response(user_vector[i], epsilon=epsilon/2)
    return user_vector


def hist_estimation_experiment():
    # 生成数据
    num_users = 100000
    dimension = 100
    index_1 = np.clip(np.random.normal(loc=50, scale=10, size=[num_users]), 0, dimension)
    users = [[0 for i in range(dimension)] for j in range(num_users)]
    for i in range(num_users):
        users[i][round(index_1[i])] = 1

    # 得到原始数据的直方图
    original_hist = np.sum(users, axis=0)
    print("this is original hist: \n", original_hist)

    # 隐私参数
    epsilon = np.log(3)

    # aggregator收集并处理数据
    rr_user = np.asarray(
        [random_response_for_hist(user, epsilon) for user in users])
    rr_sums = np.sum(rr_user, axis=0)
    print("this is the hist by the aggregator: \n", rr_sums)

    # aggregator校正数据
    estimate_hist = [random_response_adjust(
        rr_sum, len(users), epsilon/2) for rr_sum in rr_sums]
    print(np.sum(estimate_hist))

    # 展示原始数据的直方图
    print("this is estimated hist: \n", estimate_hist)

    # 画图
    fig = plt.figure(figsize=[12, 5])
    ax1 = fig.add_subplot(121)  # 2*2的图形 在第一个位置
    ax1.bar(range(len(original_hist)), original_hist)
    ax2 = fig.add_subplot(122)
    ax2.bar(range(len(estimate_hist)), estimate_hist)
    plt.show()


if __name__ == '__main__':
    # mean_estimation_experiment()
    hist_estimation_experiment()
