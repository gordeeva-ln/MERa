"""
Модель, описывающая MERa
Она определяется следующим набором параметров:
    lang: на каком языке тексты
    mode: какой режим выбора признаков:
        default: все признаки, применимые к этому языку
        black_list: множество признаков, которые надо исключить
        white_list: множество признаков, которые надо оставить
    features: список признаков, если выбран не default mode

"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from scipy.special import expit
from feature.main import FEATURES
from utils import optimize, tokenizer_split, alignment, show_scores, train_test_split, standard, unstandard_weights
from pymorphy2 import MorphAnalyzer
from nltk.stem import PorterStemmer

VECTORS = {}
CLF = LogisticRegression(multi_class="multinomial", solver="newton-cg", random_state=0, max_iter=10000, tol=1e-5)


class MERa:
    def select_features(self, features):
        selected = set()

        for feature in FEATURES:
            if self.mode == "default" \
                    or self.mode == "white_list" and feature in features \
                    or self.mode == "black_list" and feature not in features:
                selected.add(feature)
        return selected

    def __init__(self, lang, mode="default", features=set()):
        if lang not in {"russian", "english"}:
            raise Exception(f"Invalid language {lang}")
        self.lang = lang

        self.analyzer = MorphAnalyzer() if lang == "russian" else PorterStemmer()

        if mode not in ["default", "black_list", "white_list"]:
            raise Exception("Incorrect mode")
        self.mode = mode

        if features | FEATURES.keys() != FEATURES.keys():
            raise Exception(f"Following features are not defined: {(features | FEATURES.keys()) - FEATURES.keys()}")

        self.features = sorted(list(self.select_features(features)), key=lambda x: FEATURES[x].order)
        self.size = len(self.features)

        self.weights = [0, 1] + [0] * (self.size - 2)  # по умолчанию работает как WER

    def __call__(self, ref, hyp, show=False, *args, **kwargs):
        ref_words, hyp_words = tokenizer_split(ref, hyp)
        pairs, costs = optimize(ref_words, hyp_words, self.cost)
        diff_ref, diff_hyp, diff_cost = alignment(pairs, costs)

        value = sum(costs) / max(len(ref_words), len(hyp_words))
        value = expit(value)

        if show:
            print(diff_ref)
            print(diff_hyp)
            print(diff_cost)

        return value, {
            "diff_ref": diff_ref,
            "diff_hyp": diff_hyp,
            "diff_cost": diff_cost
        }

    def vector(self, word1, word2):
        """
        Вектор признаков (далее надо вызывать только если нет в кеше)
        :param word1:
        :param word2:
        :return:
        """
        if (word1, word2) in VECTORS:
            return VECTORS[(word1, word2)]
        x = [0] * self.size
        index = 0
        for feature in self.features:
            x[index] = FEATURES[feature](word1, word2, self.analyzer)
            index += 1
        VECTORS[(word1, word2)] = x
        return x

    def cost(self, word1, word2):
        return np.array(self.vector(word1, word2)).T.dot(np.array(self.weights))

    def fit(self, X_texts, y, probs):
        index_train, index_test = train_test_split(len(y))
        old_mean, old_std = np.zeros((self.size + 1,)), np.ones((self.size + 1,))
        for _ in range(10):
            X, old_mean, old_std = self.e_step(X_texts, old_mean, old_std)
            self.m_step(X[index_train], np.array(y)[index_train], np.array(probs)[index_train])
            show_scores(index_train, index_test, X, probs, self.weights)

        self.weights = unstandard_weights(X, self.weights, old_mean, old_std)

    def e_step(self, X_texts, old_mean, old_std):
        X = []
        for ref, hyp in X_texts:
            pairs, costs = optimize(*tokenizer_split(ref, hyp), self.cost)
            # alignment(pairs, costs)
            X.append(np.sum([self.vector(word1, word2) for word1, word2 in pairs], 0) / max(len(ref), len(hyp)))

        X, new_mean, new_std, self.weights = standard(np.array(X), self.weights, old_mean, old_std)
        return X, new_mean, new_std

    def m_step(self, X, y, probs):
        CLF.fit(X, y, sample_weight=[probs[i][y[i]] for i in range(len(y))])
        self.weights = CLF.coef_[0]
