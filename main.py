"""
Описание класса FEATURE
Признак описывается четырьмя полями:
    description - развернутое описание со всеми тонкостями
    expression - лямбда, принимающая на вход два слова и отдающая число
    lang - множество языков, к которому признак применим
    order - порядковый номер (удобно, для работы с весами)
"""
import Levenshtein
import numpy as np
import fasttext.util
# fasttext.util.download_model('ru')
print("downloaded")
ft = fasttext.load_model('cc.ru.300.bin')


class Feature:
    def __init__(self, description, expression, lang, order):
        self.description = description
        self.expression = expression
        self.lang = lang
        self.order = order

    def __call__(self, *args, **kwargs):
        return self.expression(*args)


FEATURES = {
    "const": Feature(
        description=
        """
        1 for all inputs
        """,
        expression=lambda word1, word2, analyzer: 1,
        lang={"russian", "english"},
        order=0
    ),
    "equals": Feature(
        description=
        """
        Проверка слов на полное равенство
        """
        ,

        expression=lambda word1, word2, analyzer:
            int(word1 == word2),

        lang={"russian", "english"},
        order=1
    ),
    "not equals": Feature(
        description=
        """
        Проверка слов на неравенство
        """
        ,

        expression=lambda word1, word2, analyzer:
            int(word1 != word2),

        lang={"russian", "english"},
        order=2
    ),
    "lemmas equals": Feature(
        description=
        """
        Проверка словарных форм слов на равенство
        """
        ,

        expression=lambda word1, word2, analyzer:
            int(analyzer.parse(word1)[0].normal_form == analyzer.parse(word2)[0].normal_form),

        lang={"russian", "english"},
        order=3
    ),
    "lemmas not equals": Feature(
        description=
        """
        Проверка словарных форм слов на неравенство
        """
        ,

        expression=lambda word1, word2, analyzer:
        int(analyzer.parse(word1)[0].normal_form != analyzer.parse(word2)[0].normal_form),

        lang={"russian"},
        order=4
    ),
    "length difference": Feature(
        description=
        """
        Модуль разности между длинами слов
        """
        ,

        expression=lambda word1, word2, analyzer:
        abs(len(word1) - len(word2)),

        lang={"russian", "english"},
        order=5
    ),
    "length difference norm by max": Feature(
        description=
        """
        Модуль разности между длинами слов разделенный на максимум из длин
        """
        ,

        expression=lambda word1, word2, analyzer:
        abs(len(word1) - len(word2)) / max(len(word1), len(word2)),

        lang={"russian", "english"},
        order=6
    ),

    "Levenshtein difference": Feature(
        description=
        """
        Редакционное расстояние между словами
        """
        ,

        expression=lambda word1, word2, analyzer:
        Levenshtein.distance(word1, word2),

        lang={"russian", "english"},
        order=7
    ),
    "Levenshtein difference (lemmas)": Feature(
        description=
        """
        Редакционное расстояние между словырными формами
        """
        ,

        expression=lambda word1, word2, analyzer:
        Levenshtein.distance(analyzer.parse(word1)[0].normal_form, analyzer.parse(word2)[0].normal_form),

        lang={"russian"},
        order=8
    ),
    "Levenshtein difference (lemmas) norm by length sum": Feature(
        description=
        """
        Редакционное расстояние между словарными формами
        """
        ,

        expression=lambda word1, word2, analyzer:
        Levenshtein.distance(analyzer.parse(word1)[0].normal_form, analyzer.parse(word2)[0].normal_form) /
        (len(analyzer.parse(word1)[0].normal_form) + len(analyzer.parse(word2)[0].normal_form)),

        lang={"russian"},
        order=9
    ),
    "FastText embeddings": Feature(
        description=
        """
        Косинусное расстояние между fasttext embeddings
        """
        ,

        expression=lambda word1, word2, analyzer: 1 - np.dot(ft.get_word_vector(word1), ft.get_word_vector(word2)),

        lang={"russian"},
        order=10
    ),
    "Reference in dictionary": Feature(
        description=
        """
        Исходное слово есть в словаре
        """
        ,

        expression=lambda word1, word2, analyzer:
        analyzer.parse(word1)[0].is_known,

        lang={"russian"},
        order=11
    ),
    "Hypothesis in dictionary": Feature(
        description=
        """
        Предсказанное слово есть в словаре
        """
        ,

        expression=lambda word1, word2, analyzer:
        analyzer.parse(word2)[0].is_known,

        lang={"russian"},
        order=12
    ),
    "E with dots": Feature(
        description=
        """
        Слова равны с точностью до замены Ё на Е.
        """
        ,

        expression=lambda word1, word2, analyzer:
        word1.replace('ё', 'е') == word2.replace('ё', 'е'),

        lang={"russian"},
        order=13
    ),
    "error in no word": Feature(
        description=
        """
        Ошибка в слове "не"
        """
        ,

        expression=lambda word1, word2, analyzer:
        word1 != word2 and "не" in {word1, word2},

        lang={"russian"},
        order=14
    ),
    "insertion": Feature(
        description=
        """
        Вставка
        """
        ,

        expression=lambda word1, word2, analyzer:
        word1 == "",

        lang={"russian", "english"},
        order=15
    ),
    "deletion": Feature(
        description=
        """
        Удаление
        """
        ,

        expression=lambda word1, word2, analyzer:
        word2 == "",

        lang={"russian", "english"},
        order=16
    ),
}


# TODO перенести все оставшиеся признаки


if __name__ == '__main__':
    print(FEATURES["equals"]("hjk", "", ))
