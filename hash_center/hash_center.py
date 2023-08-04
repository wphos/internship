import math
import random

import numpy as np


def generate_hash_center(num_of_classes, code_length):
    h = generate_matrix(code_length, num_of_classes)
    rho = 1.02
    max_mu = 1e10
    epsilon = 1e-6
    T = 50
    d = get_minimal_distance(num_of_classes, code_length)
    e = (code_length - 2 * d) * np.ones(num_of_classes - 1)

    k1, k2 = (np.zeros(code_length) for i in range(2))
    k3 = np.zeros(num_of_classes - 1)
    best_dis = -1

    for t in range(T):
        should_update_k = True
        print('t : %s' % t)
        for i in range(num_of_classes):
            print('i : %s' % i)
            v1, v2 = (np.zeros(code_length) for i in range(2))
            v3 = np.zeros(num_of_classes - 1)
            mu = 1e-6
            hi = h[:, i]
            h_subtract_i = np.delete(h, i, axis=1)  # h_{~i}

            times = 0
            while True:
                # update hi
                hi = update_hi(h_subtract_i, mu, v1, v2, v3, k1, k2, k3, code_length, e)

                # update v1, v2, v3
                v1 = update_v1(hi, k1, mu)
                v2 = update_v2(hi, k2, mu, code_length)
                v3 = update_v3(e, hi, h_subtract_i, k3, mu)

                # update k1, k2, k3
                if should_update_k:
                    print('update k')
                    k1 = update_k1(k1, mu, hi, v1)
                    k2 = update_k2(k2, mu, hi, v2)
                    k3 = update_k3(k3, mu, hi, h_subtract_i, v3, e)
                    should_update_k = False

                # update mu
                mu = min(rho * mu, max_mu)

                # determine whether to break
                tmp1 = infty_norm(hi - v1)
                tmp2 = infty_norm(hi - v2)
                tmp3 = infty_norm(hi @ h_subtract_i + v3 - e)
                if times % 5000 == 0:
                    print('classes=%s, code_length=%s, best_dis=%s, t=%s, i=%s, %s %s %s' %
                          (num_of_classes, code_length, best_dis, t, i, tmp1, tmp2, tmp3))
                if max(tmp1, tmp2, tmp3) <= epsilon or times > 100000:
                    h[:, i] = hi
                    break

                times += 1
        h[h == 0] = -1 if random.randint(0, 1) % 2 == 0 else 1
        print(h)
        cur_dis = min_hamming_distance(h)
        print('cur dis', cur_dis)
        if cur_dis > best_dis:
            best_dis = cur_dis
            np.savetxt('centers/center_%s_%s.txt' % (num_of_classes, code_length), h)
        if cur_dis >= d and np.all((h == 1) | (h == -1)):
            break
    return h


# update k1 via Eq.(11)
def update_k1(k1, mu, hi, v1):
    return k1 + mu * (hi - v1)


# update k2 via Eq.(11)
def update_k2(k2, mu, hi, v2):
    return k2 + mu * (hi - v2)


# update k3 via Eq.(11)
def update_k3(k3, mu, hi, h_subtract_i, v3, e):
    return k3 + mu * (hi @ h_subtract_i + v3 - e)


# update v1 via Eq.(10)
def update_v1(hi, k1, mu):
    tmp = hi + k1 / mu
    tmp[tmp > 1] = 1
    tmp[tmp < -1] = -1
    return tmp


# update v2 via Eq.(10)
def update_v2(hi, k2, mu, q):
    tmp = hi + k2 / mu
    tmp = tmp / np.linalg.norm(tmp)
    tmp = math.sqrt(q) * tmp
    return tmp


def update_v3(e, hi, h_subtract_i, k3, mu):
    tmp = e - hi @ h_subtract_i - k3 / mu
    tmp[tmp < 0] = 0
    return tmp


# update hi via Eq.(7)
def update_hi(h_subtract_i, mu, v1, v2, v3, k1, k2, k3, q, e):
    tmp1 = np.linalg.inv(2 * mu * np.eye(q) + mu * (h_subtract_i @ np.transpose(h_subtract_i)))

    tmp2 = mu * (v1 + v2 + h_subtract_i @ e - h_subtract_i @ v3)
    h_sum = np.sum(np.hsplit(h_subtract_i, h_subtract_i.shape[1]), axis=0).flatten()
    tmp3 = tmp2 - h_sum - k1 - k2 - h_subtract_i @ k3

    return tmp1 @ tmp3


def generate_hadamard_matrix(n):
    if n == 1:
        return np.array([[1]])

    # 生成 n/2 阶 Hadamard 矩阵
    h_n_2 = generate_hadamard_matrix(n // 2)

    # 构造 n 阶 Hadamard 矩阵
    h_n = np.block([[h_n_2, h_n_2],
                    [h_n_2, -h_n_2]])

    return h_n


def generate_matrix(row, column):
    return np.random.choice([-1, 1], size=(row, column), p=[0.5, 0.5])


def get_minimal_distance(num_of_classes: int, code_length: int):
    baseline = math.pow(2, code_length) / num_of_classes
    amount = 0
    for i in range(code_length):
        amount += math.comb(code_length, i)
        if amount >= baseline:
            return i + 1
    return 0


def infty_norm(v):
    return np.max(np.abs(v))


def hamming_distance(v1, v2):
    return np.count_nonzero(v1 != v2)


def min_hamming_distance(matrix: np.ndarray):
    num_columns = matrix.shape[1]
    min_distance = float('inf')
    for i in range(num_columns):
        for j in range(i + 1, num_columns):
            distance = hamming_distance(matrix[:, i], matrix[:, j])
            if distance < min_distance:
                min_distance = distance
    return min_distance
