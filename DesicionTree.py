import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate


class DT:
    def __init__(self, max_depth=None) -> None:
        self.data = None
        self.n_data = None
        self.n_features = None
        self.label = None
        self.n_positive = None
        self.n_negative = None
        self.distinct_value = None
        self.child: DT = None
        self.max_depth = max_depth
        self.selected_attribute = None

    def fill_objects(self, data: np.ndarray, label: np.ndarray):
        """
        Get data and label then fill related objects in DT.
        """
        self.data = data
        self.label = label

        # Get number of data
        self.n_data = self.data.shape[0]
        # Get number of features
        self.n_features = self.data.shape[1]
        # Find distinct value of each column
        self.distinct_value = [
            np.unique(self.data[:, i]) for i in range(self.n_features)
        ]
        # Get number of positive and negative label
        self.n_positive = np.sum(self.label)
        self.n_negative = self.n_data - self.n_positive

    def n_v_pv_nv(self, column_id, value):
        """
        Return total number of data which has value in column_id
        and number of positive and negative label.
        """

        # Get label of data which has value in column_id
        label = self.label[self.data[:, column_id] == value]

        return label.shape[0], np.sum(label), label.shape[0] - np.sum(label)

    def calc_hellinger_distance(self, column_id: int):
        """
        Calculate Hellinger distance of a column.
        """

        # Get distinct value of column
        distinct_value = self.distinct_value[column_id]

        # Initialize Hellinger distance
        hellinger = 0

        # Calculate Hellinger distance
        for value in distinct_value:
            # Get label of data which has value in column_id
            (
                n_value,
                n_value_positive_label,
                n_value_negative_label,
            ) = self.n_v_pv_nv(column_id, value)

            hellinger += (
                np.sqrt(n_value_positive_label / n_value)
                - np.sqrt(n_value_negative_label / n_value)
            ) ** 2

        return np.sqrt(hellinger)

    def attribute_selection(self):
        """
        Select attribute to split data.
        """

        # Initialize Hellinger distance
        max_hellinger = 0

        # Initialize column id
        max_hellinger_id = 0

        # Calculate Hellinger distance of each column
        for i in range(self.n_features):
            # Calculate Hellinger distance of column i
            hellinger_i = self.calc_hellinger_distance(column_id=i)

            # Update Hellinger distance and column id
            if hellinger_i > max_hellinger:
                max_hellinger = hellinger_i
                max_hellinger_id = i

        return max_hellinger_id

    def split_child(self, selected_attribute: int):
        """
        Split data by selected_attribute and return child.
        """

        # Get distinct value of selected_attribute
        distinct_value = self.distinct_value[selected_attribute]

        # Initialize child
        child = {}

        # Split data by selected_attribute
        for value in distinct_value:
            data_child, label_child = self.child_data(
                selected_attribute, value
            )
            child[value] = self.fit_child(data_child, label_child)

        return child

    def child_data(self, selected_attribute: int, value: int):
        """
        Retrurn data and label which has value in selected_attribute
          and remove selected_attribute.
        """
        # Get data which has value in selected_attribute
        #  and remove selected_attribute
        data = np.delete(
            self.data[self.data[:, selected_attribute] == value],
            selected_attribute,
            1,
        )

        # Get label of data which has value in column_id
        label = self.label[self.data[:, selected_attribute] == value]

        return data, label

    def fit_child(self, data, label):
        """
        Create subtree.
        """

        # Create subtree
        subtree = DT()
        if self.max_depth is not None:
            subtree.max_depth = self.max_depth - 1

        subtree.fit(data, label)

        return subtree

    def fit(self, data, label):
        """
        Fit data to model.
        """

        self.fill_objects(data, label)

        # Check if tree is leaf
        if self.max_depth == 0 or self.n_features == 1:
            return

        # Select attribute to split data
        self.selected_attribute = self.attribute_selection()

        # Split data by selected attribute
        self.child = self.split_child(self.selected_attribute)

    def predict_sample(self, sample):
        """
        Predict label of sample.
        """

        # Check if tree is leaf
        if self.max_depth == 0 or self.n_features == 1:
            return self.n_positive >= self.n_negative

        # Get value of selected_attribute
        value = sample[self.selected_attribute]

        # Check if value is not in distinct_value
        if value not in self.distinct_value[self.selected_attribute]:
            return self.n_positive >= self.n_negative

        # Predict label of sample
        return self.child[value].predict_sample(
            np.delete(sample, self.selected_attribute)
        )

    def predict(self, data: np.ndarray):
        predicted = []
        for sample in data:
            predicted.append(self.predict_sample(sample))

        return np.array(predicted)


def read_dataset(file_path: str):
    """
    Read dataset from file_path which is a csv file.
    """
    # Read data from file_path
    dataset = pd.read_csv(file_path, sep=",")
    # Get data from data and convert to numpy array
    data = dataset.iloc[:, :-1].to_numpy().astype(int)
    # Get label from data and convert to numpy array
    label = dataset.iloc[:, -1].to_numpy().astype(bool)

    return data, label


def accuracy(predict, label):
    """
    Calculate accuracy of predict.
    """
    # Precision
    precision = np.sum(predict == label) / predict.shape[0]
    # Recall
    recall = np.sum(predict == label) / label.shape[0]
    # F-measure
    f_measure = 2 * precision * recall / (precision + recall)
    # AUC
    auc = roc_auc_score(label, predict)
    # G-mean
    g_mean = np.sqrt(precision * recall)

    return np.array([precision, recall, f_measure, auc, g_mean])


def homework(
    DS_path: str,
    n_iteration: int,
    max_depth: int = None,
):
    data, label = read_dataset(DS_path)
    acc = np.empty((0, 5))

    for _ in range(n_iteration):
        # Split data to train and test with sklearn
        data_train, data_test, label_train, label_test = train_test_split(
            data, label, test_size=0.3
        )

        # Create model
        model = DT(max_depth=max_depth)
        model.fit(data_train, label_train)
        predicted = model.predict(data_test)
        acc = np.vstack((acc, accuracy(predicted, label_test)))

    label_table = ["Precision", "Recall", "F-measure", "AUC", "G-mean"]
    print(
        tabulate(
            acc,
            headers=label_table,
            tablefmt="fancy_grid",
            showindex="always",
            floatfmt=".4f",
        )
    )
    for metrinc in label_table:
        print(
            f"\navg {metrinc}:",
            f"{np.mean(acc[:, label_table.index(metrinc)]).round(4)}",
        )


homework(DS_path="Dataset/Covid19HDDT.csv", max_depth=1, n_iteration=10)
