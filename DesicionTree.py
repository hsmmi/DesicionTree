import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from tabulate import tabulate


class metric:
    def n_v_pv_nv(x: np.ndarray, y: np.ndarray, value):
        """
        Return number of positive and negative label.
        """

        # Get label of data which has value in column_id
        label = y[x == value]

        return np.sum(label), label.shape[0] - np.sum(label)

    def calc_hellinger_distance(x: np.ndarray, y: np.ndarray):
        """
        Calculate Hellinger distance of a column and label.
        """

        # Get number of positive and negative label
        n_positive_label = np.sum(y)
        n_negative_label = x.shape[0] - n_positive_label

        # Initialize Hellinger distance
        hellinger = 0

        # Calculate Hellinger distance
        for value in np.unique(x):
            # Get label of data which has value in column_id
            (
                n_value_positive_label,
                n_value_negative_label,
            ) = metric.n_v_pv_nv(x, y, value)

            hellinger += (
                np.sqrt(n_value_positive_label / n_positive_label)
                - np.sqrt(n_value_negative_label / n_negative_label)
            ) ** 2

        return np.sqrt(hellinger)


class preprocess:
    def read_dataset(file_path: str):
        """
        Read dataset from file_path which is a csv file and return data and
        label.
        """
        # Read data from file_path
        dataset = pd.read_csv(file_path, sep=",")
        # Get data from data and convert to numpy array
        data = dataset.iloc[:, :-1].to_numpy().astype(int)
        # Get label from data and convert to numpy array
        label = dataset.iloc[:, -1].to_numpy().astype(int)

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

    def remove_correlated_with_label_by_hellinger(
        data: np.ndarray, label: np.ndarray, threshold: float = 0.4
    ):
        """
        Remove features which are correlated with label more than threshold.
        """

        # Get number of features
        n_features = data.shape[1]

        # Initialize correlation
        correlation = np.zeros(n_features)

        # Calculate correlation of each feature with label
        for i in range(n_features):
            correlation[i] = metric.calc_hellinger_distance(data[:, i], label)

        # Get index of features which are correlated with label more than
        # threshold
        index = np.where(np.abs(correlation) > threshold)[0]

        # Remove features which are correlated with label more than threshold
        data = np.delete(data, index, axis=1)

        return data

    def minority_0_majority_1(label: np.ndarray):
        """
        Change minority class to 0 and majority class to 1.
        """
        label = label.copy()

        # Find count of each unique label
        uniqe_label, count_label = np.unique(label, return_counts=True)
        # Get label of minority class
        label_minority = uniqe_label[count_label.argmin()]
        # Change minority class to -1 for be seprated from others
        label[label == label_minority] = -1
        # Change other classes to 1
        label[label != -1] = 1
        # Change minority class to 0
        label[label == -1] = 0

        return label

    def OVO(data: np.ndarray, label: np.ndarray, l1: int, l2: int):
        """
        One vs One.
        """
        data = data.copy()
        label = label.copy()
        # Get data which has label l1 or l2
        data = data[(label == l1) | (label == l2)]
        # Get label which has label l1 or l2
        label = label[(label == l1) | (label == l2)]

        # Change label to 0 or 1
        label[label == l1] = 0
        label[label == l2] = 1

        return data, label

    def OVA(label: np.ndarray, l_sel: int):
        """
        One vs All.
        """
        label = label.copy()
        # Change our class to -1 to change all remaining to 0
        label[label == l_sel] = -1
        # Cahnge all other class to 0
        label[label != -1] = 0
        # Change selected class to 1
        label[label == l_sel] = 1

        return label


class DT:
    def __init__(self, max_depth=None, cut_off_size=None) -> None:
        self.data = None
        self.n_data = None
        self.n_features = None
        self.label = None
        self.n_positive = None
        self.n_negative = None
        self.distinct_value = None
        self.child: DT = None
        self.max_depth = max_depth
        self.cut_off_size = cut_off_size
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
        # Get number of positive and negative label
        self.n_positive = np.sum(self.label)
        self.n_negative = self.n_data - self.n_positive

    def n_v_pv_nv(self, x: np.ndarray, y: np.ndarray, value):
        """
        Return number of positive and negative label.
        """

        # Get label of data which has value in column_id
        label = y[x == value]

        return np.sum(label), label.shape[0] - np.sum(label)

    def calc_hellinger_distance(self, x: np.ndarray, y: np.ndarray):
        """
        Calculate Hellinger distance of a column and label.
        """

        # Get number of positive and negative label
        n_positive_label = np.sum(y)
        n_negative_label = x.shape[0] - n_positive_label

        # Initialize Hellinger distance
        hellinger = 0

        # Calculate Hellinger distance
        for value in np.unique(x):
            # Get label of data which has value in column_id
            (
                n_value_positive_label,
                n_value_negative_label,
            ) = self.n_v_pv_nv(x, y, value)

            hellinger += (
                np.sqrt(n_value_positive_label / n_positive_label)
                - np.sqrt(n_value_negative_label / n_negative_label)
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
            hellinger_i = self.calc_hellinger_distance(
                self.data[:, i], self.label
            )

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
        distinct_value = np.unique(self.data[:, selected_attribute])
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

        # Check cut-off size
        if self.cut_off_size is not None and self.n_data < self.cut_off_size:
            return

        # Check purity
        if self.n_negative == 0 or self.n_positive == 0:
            return

        # Select attribute to split data
        self.selected_attribute = self.attribute_selection()

        # Split data by selected attribute
        self.child = self.split_child(self.selected_attribute)

        self.remove_data()

    def remove_data(self):
        """
        remove data and label.
        """

        self.data = None
        self.label = None

    def predict_sample(self, sample):
        """
        Predict label of sample.
        """

        # Check if tree is leaf
        if self.selected_attribute is None:
            return self.n_positive >= self.n_negative

        # Get value of selected_attribute
        value = sample[self.selected_attribute]

        # Check if value is not in distinct_value
        if value not in self.child.keys():
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

    def run_HDDT(
        data: np.ndarray,
        label: np.ndarray,
        n_iteration: int,
        max_depth: int = None,
        cut_off: int = None,
        print_tabel: bool = False,
    ):
        acc = np.empty((0, 5))
        for _ in range(n_iteration):
            # Split data to train and test with sklearn
            data_train, data_test, label_train, label_test = train_test_split(
                data, label, test_size=0.3, stratify=label
            )

            # Create model
            model = DT(max_depth=max_depth, cut_off_size=cut_off)
            model.fit(data_train, label_train)
            predicted = model.predict(data_test)
            acc = np.vstack((acc, preprocess.accuracy(predicted, label_test)))

        label_table = ["Precision", "Recall", "F-measure", "AUC", "G-mean"]
        if print_tabel:
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
                f"avg {metrinc}:",
                f"{np.mean(acc[:, label_table.index(metrinc)]).round(4)}",
            )


DS_path = "Dataset/Covid19HDDT.csv"

max_depth = [2, 3, 4, 5]
cut_off = [10, 50, 500]
n_iteration = 10

data, label = preprocess.read_dataset(DS_path)

label_minority_majority = preprocess.minority_0_majority_1(label)
# Remove most correlatied feature from data
data = preprocess.remove_correlated_with_label_by_hellinger(
    data=data, label=label_minority_majority, threshold=0.3
)

DT.run_HDDT(
    data,
    label_minority_majority,
    max_depth=2,
    n_iteration=n_iteration,
)
