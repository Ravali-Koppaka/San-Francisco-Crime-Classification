import numpy as np


def entropy(attribute_data):
    value, value_count = np.unique(attribute_data, return_counts=True)
    value_proportion = value_count / len(attribute_data)
    return -value_proportion.dot(np.log(value_proportion) / np.log(2))


def info_gain(attribute_data, labels):
    attr_val_counts = get_count_dict(attribute_data)
    total_count = len(labels)
    entropy_val = 0.0
    for attr_val, attr_val_count in attr_val_counts.items():
        entropy_val += attr_val_count * entropy(labels[attribute_data == attr_val])

    info = entropy(labels) - entropy_val / total_count
    return info


def get_count_dict(data):
    data_values, data_count = np.unique(data, return_counts=True)
    return dict(zip(data_values, data_count))


def find_majority_label(data):
    value, value_count = np.unique(data, return_counts=True)
    max_value = 0
    index = 0
    for i in range(len(value_count)):
        if value_count[i] > max_value:
            max_value = value_count[i]
            index = i
    return value[index]


class DecisionStump:
    best_attribute = None
    children = None

    def choose_best_attribute(self, data, labels):
        best_gain = float('-inf')
        attributes = data.keys()
        for attr in attributes:
            attribute_data = data[attr]
            gain = info_gain(attribute_data, labels)
            if gain > best_gain:
                best_gain = gain
                self.best_attribute = attr
        return

    def fit(self, data, labels):
        if data.size == 0:
            return

        self.choose_best_attribute(data, labels)

        attribute_data = data[self.best_attribute]

        self.children = []
        for val in np.unique(attribute_data):
            child_labels = labels[attribute_data == val]
            self.children.append((val, find_majority_label(child_labels)))

    def predict(self, data):
        if data.size == 0:
            return

        labels = np.zeros(len(data)).tolist()

        for child in self.children:
            index = np.where(data[self.best_attribute] == child[0])
            index_array = index[0].tolist()
            for i in range(len(index_array)):
                labels[index_array[i]] = child[1]
        return np.asarray(labels)
