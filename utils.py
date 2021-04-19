import numpy as np
from sklearn.utils import resample
import random
from catboost.utils import eval_metric
from scipy.special import expit


def tokenizer_split(ref, hyp):
    return ref.split(), hyp.split()


def optimize(words_ref, words_hyp, cost):
    # --> define dynamic table size
    n = len(words_ref)
    m = len(words_hyp)

    # --> create table
    table = np.full((n + 1, m + 1), np.inf)
    table[0, 0] = 0

    # --> fill table
    for i in range(n):
        for j in range(m):
            # --> deletion
            table[i + 1, j] = min(table[i + 1][j], table[i][j] + cost(words_ref[i], ""))

            # --> insertion
            table[i, j + 1] = min(table[i][j + 1], table[i][j] + cost("", words_hyp[j]))

            # --> substitution
            table[i + 1, j + 1] = min(table[i + 1][j + 1],
                                      table[i][j] + cost(words_ref[i], words_hyp[j]))

    # --> fill table edge
    for j in range(m):
        # --> only insertion possible
        table[n, j + 1] = min(table[n][j + 1], table[n][j] + cost("", words_hyp[j]))

    for i in range(n):
        # --> deletion
        table[i + 1, m] = min(table[i + 1][m], table[i][m] + cost(words_ref[i], ""))

    ### RECONSTRUCTION ###
    eps = 1e-10
    pairs = []
    costs = []
    i, j = n, m

    while i > 0 or j > 0:
        new_i, new_j = i, j

        if i > 0:
            # --> deletion
            if abs(table[i - 1, j] + cost(words_ref[i - 1], "") - table[i, j]) < eps:
                pair = (words_ref[i - 1], "")
                new_i, new_j = i - 1, j

        if i > 0 and j > 0:
            # --> substitution
            if abs(table[i - 1, j - 1] + cost(words_ref[i - 1], words_hyp[j - 1]) - table[i, j]) < eps:
                pair = (words_ref[i - 1], words_hyp[j - 1])
                new_i, new_j = i - 1, j - 1

        if j > 0:
            # --> insertion
            if abs(table[i, j - 1] + cost("", words_hyp[j - 1]) - table[i, j]) < eps:
                pair = ("", words_hyp[j - 1])
                new_i, new_j = i, j - 1

        pairs.append(pair)
        costs.append(cost(pair[0], pair[1]))
        i, j = new_i, new_j

    # --> reverse pairs
    pairs.reverse()
    costs.reverse()

    return pairs, costs


def alignment(pairs, costs):

    diff_ref = []
    diff_hyp = []
    diff_cost = []

    for i in range(len(costs)):
        ref, hyp = pairs[i]
        cost = str(round(costs[i], 3))

        length = max(len(ref), len(hyp), len(cost))
        diff_ref.append(ref + "*" * (length - len(ref)))
        diff_hyp.append(hyp + "*" * (length - len(hyp)))
        diff_cost.append(cost + " " * (length - len(cost)))

    return ' '.join(diff_ref), ' '.join(diff_hyp), ' '.join(diff_cost)


def train_test_split(size, p=0.6):
    indexes = list(range(size))
    random.shuffle(indexes)
    k = round(size * p)
    return indexes[:k], indexes[k:]


def show_scores(index_train, index_test, X, probs, weights):
    auc_test = bootstrap(1000, 0.75,
                         np.array(probs)[:, 1][index_test],
                         expit(X[index_test].dot(np.array(weights))),
                         lambda a1, a2: eval_metric(a1, a2, 'AUC'))  # на классе 1
    auc_train = bootstrap(1000, 0.75,
                          np.array(probs)[:, 1][index_train],
                          expit(X[index_train].dot(np.array(weights))),
                          lambda a1, a2: eval_metric(a1, a2, 'AUC'))
    print("Standard weights\n")
    print(weights)
    # feature_weights_table(features, weights)
    print("AUC test", auc_test)
    print("AUC train", auc_train)


def bootstrap(k, p, seq1, seq2, func):
    """

    :param k: samples count
    :param p: samples part
    :param seq1, seq2: two list for comparing
    :param func: score function
    :return: interval
    """

    values = []
    n = round(len(seq1) * p)
    for i in range(k):
        indexes = resample(list(range(len(seq1))), n_samples=n)
        values.append(func(seq1[indexes], seq2[indexes]))
    values.sort()
    tails = int(0.025 * len(values))
    return values[tails], values[-tails]


def feature_weights_table(first_column, second_column):
    first_column_size = max([len(row) for row in first_column])
    for i in range(len(first_column)):
        print(f'{first_column[i]}{" " * (first_column_size - len(first_column[i]))}: {round(second_column[i], 4)}')


def standard(x, weights, old_mean, old_std):
    new_mean = np.mean(x, axis=0)
    new_std = np.std(x, axis=0)
    # old_mean = np.mean(X, axis=0)
    # old_std = np.std(X, axis=0)

    x_std = x - new_mean
    x_std /= new_std

    # --> consts
    for i in range(1, x.shape[1]):
        if not new_std[i] or not old_std[i] or (set(x[:, i]) | {0, 1} == {0, 1}):
            x_std[:, i] = x[:, i]  # np.zeros((x.shape[0], ))
        # --> обновляем веса с учетом стандартизации
        else:
            # --> обновляем константу
            weights[0] -= weights[i] * (old_mean[i] / old_std[i] - new_mean[i] / old_std[i])

            # --> обновляем вес, соответсвующий признаку
            weights[i] *= new_std[i] / old_std[i]

    x_std[:, 0] = np.ones((x.shape[0],))
    return x_std, new_mean, new_std, weights


def unstandard_weights(x, weights, old_mean, old_std):
    for i in range(1, weights.shape[0]):
        if old_std[i] and (set(x[:, i]) | {0, 1} != {0, 1}):
            # --> обновляем константу
            weights[0] -= weights[i] * old_mean[i] / old_std[i]

            # --> обновляем вес, соответсвующий признаку
            weights[i] /= old_std[i]

    return weights
