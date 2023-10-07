import numpy as np
from collections import Counter

def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    # print(target_vector)
    feature_vector = np.array(feature_vector)
    target_vector = np.array(target_vector)
    feature_vector_sort_indexes = np.argsort(feature_vector)
    thresholds = feature_vector[feature_vector_sort_indexes]
    thresholds = (thresholds[1:] + thresholds[:-1]) / 2

    R_size = len(target_vector)
    R_l_size = np.arange(1, R_size)
    R_r_size = R_size - R_l_size

    first_meet = np.unique(feature_vector[feature_vector_sort_indexes], return_index=True)[1]
    last_meet = np.append(first_meet[1:] - 1, R_size - 1)
    real_algo_indexes = last_meet[:-1]

    # print(feature_vector_sort_indexes)
    # print(target_vector)
    R_l_targets_sum = np.cumsum(target_vector[feature_vector_sort_indexes[:-1]])
    R_r_targets_sum = np.sum(target_vector) - R_l_targets_sum

    p_1_l_square = (R_l_targets_sum / R_l_size) ** 2
    p_0_l_square = (1 - R_l_targets_sum / R_l_size) ** 2
    p_1_r_square = (R_r_targets_sum / R_r_size) ** 2
    p_0_r_square = (1 - R_r_targets_sum / R_r_size) ** 2

    H_l = 1 - p_1_l_square - p_0_l_square
    H_r = 1 - p_1_r_square - p_0_r_square

    ginis = (-R_l_size / R_size * H_l - R_r_size / R_size * H_r)[real_algo_indexes]
    ginis_best_index = np.argmax(ginis)
    threshold_best = thresholds[ginis_best_index]
    ginis_best = ginis[ginis_best_index]

    return thresholds[real_algo_indexes], ginis, threshold_best, ginis_best

class DecisionTree:
    def __init__(self, feature_types, max_depth=None, min_samples_split=None, min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node):
        print(sub_y)
        if np.all(sub_y != sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        for feature in range(sub_X.shape[1]):
            feature_type = self._feature_types[feature]
            categories_map = {}

            if feature_type == "real":
                feature_vector = sub_X[:, feature]
            elif feature_type == "categorical":
                counts = Counter(sub_X[:, feature])
                clicks = Counter(sub_X[sub_y == 1, feature])
                ratio = {}
                for key, current_count in counts.items():
                    if key in clicks:
                        current_click = clicks[key]
                    else:
                        current_click = 0
                    ratio[key] = current_click / current_count
                sorted_categories = list(map(lambda x: x[1], sorted(ratio.items(), key=lambda x: x[1])))
                categories_map = dict(zip(list(range(len(sorted_categories))), sorted_categories))
                # print(categories_map)
                feature_vector = np.array(list(map(lambda x: categories_map[x], sub_X[:, feature])))
            else:
                raise ValueError
            # print(feature_vector)
            if len(feature_vector) == 3:
                continue

            _, _, threshold, gini = find_best_split(feature_vector, sub_y)
            if gini_best is None or gini > gini_best:
                feature_best = feature
                gini_best = gini
                split = feature_vector < threshold

                if feature_type == "real":
                    threshold_best = threshold
                elif feature_type == "categorical":
                    threshold_best = list(map(lambda x: x[0],
                                              filter(lambda x: x[1] < threshold, categories_map.items())))
                else:
                    raise ValueError

        if feature_best is None:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"])
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"])

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']

        feature = node["feature_split"]
        if self._feature_types[feature] == "real":
            if x[feature] < node["threshold"]:
                return self._predict_node(x, node["left_child"])
            return self._predict_node(x, node["right_child"])
        if x[feature] < node["feature_split"]:
            return self._predict_node(x, node["left_child"])
        return self._predict_node(x, node["right_child"])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

types = ['real', 'categorical', 'real', 'categorical', 'categorical']
X_train = '0.015696372114428918 2.0 0.12471295376821585 1.0 0.0 -2.2426849541854055 1.0 -0.32279480560829565 2.0 0.0 1.150035724719818 2.0 0.8416747129961416 2.0 1.0 0.9919460223426778 0.0 2.390960515463033 1.0 0.0 0.9533241281124304 2.0 0.07619958783723642 2.0 1.0 -2.0212548201949705 2.0 -0.5664459304649568 0.0 2.0 -0.334077365808097 0.0 0.036141936684072715 1.0 1.0 0.002118364683486495 1.0 -2.0749776006900293 0.0 1.0 0.405453411570191 1.0 0.24779219974854666 1.0 2.0 0.2890919409800353 0.0 -0.8971567844396987 2.0 2.0'
y_train = '1 1 0 0 1 0 0 0 0 0'
X_train = np.fromstring(X_train, sep=' ').reshape((-1, 5))
y_train = np.fromstring(y_train, sep=' ')
X_test = '0.9533241281124304 2.0 -0.32279480560829565 0.0 0.0 0.2890919409800353 0.0 -2.0749776006900293 1.0 1.0 -2.0212548201949705 2.0 2.390960515463033 0.0 2.0 1.150035724719818 0.0 0.12471295376821585 1.0 1.0 -0.334077365808097 0.0 -0.8971567844396987 2.0 2.0 0.405453411570191 1.0 0.8416747129961416 1.0 1.0 0.9919460223426778 1.0 0.24779219974854666 1.0 0.0 -2.2426849541854055 1.0 0.07619958783723642 2.0 1.0'
X_test = np.fromstring(X_test, sep=' ').reshape((-1, 5))

clf = DecisionTree(types)
clf.fit(X_train, y_train)
print(clf.predict(X_test))