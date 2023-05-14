import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.stats import mode
from sklearn.tree import DecisionTreeClassifier
import sklearn.metrics as metrics

# from sklearn.ensemble import AdaBoostClassifier
# from tabulate import tabulate


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
    def read_dataset(DS_path: str):
        """
        Read dataset from DS_path which is a csv file and return data
        and label.
        """
        # Read data from DS_path
        dataset = pd.read_csv(DS_path, sep=",")
        # Get data from data and convert to numpy array
        data = dataset.iloc[:, :-1].to_numpy()
        # Get label from data and convert to numpy array
        label = dataset.iloc[:, -1].to_numpy().astype(int)

        return data, label

    def accuracy(predict, label):
        """
        Calculate accuracy of predict.
        """

        # Precision with confusion matrix
        precision = metrics.precision_score(label, predict, average="macro")
        # Recall
        recall = metrics.recall_score(label, predict, average="macro")
        # F-measure
        f_measure = 2 * precision * recall / (precision + recall)
        # G-mean
        g_mean = np.sqrt(precision * recall)

        return np.array([precision, recall, f_measure, g_mean])

    def remove_correlated_with_label_by_hellinger(
        data: np.ndarray, label: np.ndarray, threshold: float = 1
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

    def OVO(data: np.ndarray, label: np.ndarray, class_i: int, class_j: int):
        """
        One vs One.
        class_i to 0 and class_j to 1.
        """
        data = data.copy()
        label = label.copy()
        # Get data which has label class_i or class_j
        data = data[(label == class_i) | (label == class_j)]
        # Get label which has label class_i or class_j
        label = label[(label == class_i) | (label == class_j)]

        # Change label to 0 or 1
        label[label == class_i] = 0
        label[label == class_j] = 1

        return data, label

    def OVA(label: np.ndarray, class_i: int):
        """
        One vs All.
        """
        label = label.copy()
        # Change our class to -1 to change all remaining to 0
        label[label == class_i] = -1
        # Cahnge all other class to 0
        label[label != -1] = 0
        # Change selected class to 1
        label[label == -1] = 1

        return label

    def fill_missing_value(data: np.ndarray, method: str = "mean"):
        """
        Fill missing value of data.
        """
        data = data.copy()

        # Get number of features
        n_features = data.shape[1]

        # Fill missing value of data
        for i in range(n_features):
            # Get index of data which has missing value(nan)
            index = np.where(np.isnan(data[:, i]))[0]
            # Fill missing value of data
            if method == "mean":
                data[index, i] = np.nanmean(data[:, i])
            elif method == "median":
                data[index, i] = np.nanmedian(data[:, i])
            elif method == "mode":
                data[index, i] = np.nanmedian(data[:, i])
            elif method == "random":
                data[index, i] = np.random.choice(data[:, i], index.shape[0])

        return data

    def bootstrap_with_under_sampling(data: np.ndarray, label: np.ndarray):
        """
        Under sampling.
        """
        data = data.copy()
        label = label.copy()

        # Get number of each class
        classes, count = np.unique(label, return_counts=True)

        # Get number of classes
        n_classes = len(classes)

        # Get index of each class
        index = [np.where(label == class_i)[0] for class_i in classes]
        # shuffle index of each class
        for i in range(n_classes):
            np.random.shuffle(index[i])

        # Get number of each class after under sampling
        count = count.min()

        # Initialize new data and label
        new_data = np.zeros((count * n_classes, data.shape[1]))
        new_label = np.zeros(count * n_classes)

        # Under sampling
        for i in range(n_classes):
            new_data[count * i : count * (i + 1)] = data[index[i]][:count]
            new_label[count * i : count * (i + 1)] = label[index[i]][:count]

        return new_data, new_label


class HDDT:
    def __init__(self, max_depth=None, cut_off_size=None) -> None:
        self.data = None
        self.n_data = None
        self.n_features = None
        self.label = None
        self.n_positive = None
        self.n_negative = None
        self.distinct_value = None
        self.child: HDDT = None
        self.max_depth = max_depth
        self.cut_off_size = cut_off_size
        self.selected_attribute = None

    def fill_objects(self, data: np.ndarray, label: np.ndarray):
        """
        Get data and label then fill related objects in HDDT.
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
        subtree = HDDT()
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
        Predict probability of sample being class 1.
        """

        # Check if tree is leaf
        if self.selected_attribute is None:
            return self.n_positive / self.n_data

        # Get value of selected_attribute
        value = sample[self.selected_attribute]

        # Check if value is not in distinct_value
        if value not in self.child.keys():
            return self.n_positive / self.n_data

        # Predict label of sample
        return self.child[value].predict_sample(
            np.delete(sample, self.selected_attribute)
        )

    def predict(self, data: np.ndarray):
        predicted = []
        for sample in data:
            predicted.append(int(self.predict_sample(sample) > 0.5))

        return np.array(predicted)

    def predict_probability(self, data: np.ndarray):
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
    ) -> dict:
        acc_metrics = np.empty((0, 4))
        for _ in range(n_iteration):
            # Split data to train and test with sklearn
            data_train, data_test, label_train, label_test = train_test_split(
                data, label, test_size=0.3, stratify=label
            )

            # Create model
            model = HDDT(max_depth=max_depth, cut_off_size=cut_off)
            model.fit(data_train, label_train)
            predicted = model.predict(data_test)
            acc_metrics = np.vstack(
                (acc_metrics, preprocess.accuracy(predicted, label_test))
            )

        avg_acc_metrics = np.mean(acc_metrics, axis=0).round(5)
        return {
            "Precision": avg_acc_metrics[0],
            "Recall": avg_acc_metrics[1],
            "F-measure": avg_acc_metrics[2],
            "G-mean": avg_acc_metrics[3],
        }

    def OVO(
        data: np.ndarray,
        label: np.ndarray,
        n_iteration: int,
        max_depth: int = None,
        cut_off: int = None,
    ):
        acc_metrics = np.empty((0, 4))
        for _ in range(n_iteration):
            # Split data to train and test with sklearn
            data_train, data_test, label_train, label_test = train_test_split(
                data, label, test_size=0.3, stratify=label
            )

            # Create model
            model = HDDT(max_depth=max_depth, cut_off_size=cut_off)
            n_class = np.unique(label_train).shape[0]
            predicted = np.empty((label_test.shape[0], 0))
            for i in range(n_class):
                for j in range(i + 1, n_class):
                    # Get data which has label i or j
                    data_ij, label_ij = preprocess.OVO(
                        data=data_train,
                        label=label_train,
                        class_i=i,
                        class_j=j,
                    )

                    # Create model
                    model.fit(data_ij, label_ij)

                    # Predict label
                    predicted_ij = model.predict(data_test)
                    # relabel predicted_ij
                    predicted_ij[predicted_ij == 0] = -1
                    predicted_ij[predicted_ij == 1] = j
                    predicted_ij[predicted_ij == -1] = i

                    # Add predicted_ij to predicted
                    predicted = np.hstack(
                        (predicted, predicted_ij.reshape((-1, 1)))
                    )

            # Get label of predicted by selecting most frequent(voating)
            predicted_label = mode(predicted, axis=1, keepdims=True)[
                0
            ].reshape(-1)

            acc_metrics = np.vstack(
                (acc_metrics, preprocess.accuracy(predicted_label, label_test))
            )

        avg_acc_metrics = np.mean(acc_metrics, axis=0).round(5)
        return {
            "Precision": avg_acc_metrics[0],
            "Recall": avg_acc_metrics[1],
            "F-measure": avg_acc_metrics[2],
            "G-mean": avg_acc_metrics[3],
        }

    def OVA(
        data: np.ndarray,
        label: np.ndarray,
        n_iteration: int,
        max_depth: int = None,
        cut_off: int = None,
    ):
        acc_metrics = np.empty((0, 4))
        for _ in range(n_iteration):
            # Split data to train and test with sklearn
            data_train, data_test, label_train, label_test = train_test_split(
                data, label, test_size=0.3, stratify=label
            )

            # Create model
            model = HDDT(max_depth=max_depth, cut_off_size=cut_off)
            n_class = np.unique(label_train).shape[0]
            predicted = np.empty((label_test.shape[0], 0))
            for i in range(n_class):
                # Get data which has label i
                label_i = preprocess.OVA(label=label_train, class_i=i)

                # Create model
                model.fit(data_train, label_i)

                # Predict label
                predicted_i = model.predict_probability(data_test)

                # Add predicted_i to predicted
                predicted = np.hstack(
                    (predicted, predicted_i.reshape((-1, 1)))
                )

            # Get label of predicted by selecting most frequent(voating)
            predicted_label = np.argmax(predicted, axis=1)

            acc_metrics = np.vstack(
                (acc_metrics, preprocess.accuracy(predicted_label, label_test))
            )

        avg_acc_metrics = np.mean(acc_metrics, axis=0).round(5)
        return {
            "Precision": avg_acc_metrics[0],
            "Recall": avg_acc_metrics[1],
            "F-measure": avg_acc_metrics[2],
            "G-mean": avg_acc_metrics[3],
        }


class bagging:
    def __init__(self, T: int, base_learner: str = "HDDT") -> None:
        self.T = T
        self.data = None
        self.label = None
        self.base_learner = base_learner
        self.classifiers = []
        pass

    def fit(self, data: np.ndarray, label: np.ndarray) -> None:
        self.data = data
        self.label = label

        for i in range(self.T):
            if self.base_learner == "HDDT":
                model = HDDT(max_depth=2, cut_off_size=100)
                data_i, label_i = data, label
            else:
                model = DecisionTreeClassifier()
                # Bootstrap data
                data_i, label_i = preprocess.bootstrap_with_under_sampling(
                    data=self.data, label=self.label
                )
            model.fit(data_i, label_i)
            self.classifiers.append(model)

    def predict(self, data: np.ndarray) -> np.ndarray:
        predicted = np.empty((data.shape[0], 0))
        for classifiers in self.classifiers:
            predicted = np.hstack(
                (predicted, classifiers.predict(data).reshape((-1, 1)))
            )
        predicted_label = mode(predicted, axis=1, keepdims=True)[0].reshape(-1)
        return predicted_label

    def run_bagging(
        self, data: np.ndarray, label: np.ndarray, n_iteration: int
    ):
        acc_metrics = np.empty((0, 4))
        for _ in range(n_iteration):
            # Split data to train and test with sklearn
            data_train, data_test, label_train, label_test = train_test_split(
                data, label, test_size=0.3, stratify=label
            )

            # Create model
            self.fit(data_train, label_train)
            predicted = self.predict(data_test)
            acc_metrics = np.vstack(
                (acc_metrics, preprocess.accuracy(predicted, label_test))
            )

        avg_acc_metrics = np.mean(acc_metrics, axis=0).round(5)
        std_acc_metrics = np.std(acc_metrics, axis=0).round(5)
        return {
            "Avg-Precision": avg_acc_metrics[0],
            "Avg-Recall": avg_acc_metrics[1],
            "Avg-F-measure": avg_acc_metrics[2],
            "Avg-G-mean": avg_acc_metrics[3],
        }, {
            "Std-Precision": std_acc_metrics[0],
            "Std-Recall": std_acc_metrics[1],
            "Std-F-measure": std_acc_metrics[2],
            "Std-G-mean": std_acc_metrics[3],
        }


class adaboost:
    def __init__(self, T: int):
        self.T = T
        self.w = []
        self.n_data = None
        self.classifier = []
        self.alpha = []

    def compute_error(self, label, predict):
        return np.sum(self.w * (label != predict)) / np.sum(self.w)

    def compute_alpha(self, err):
        return np.log((1 - err) / err)

    def update_weights(self, label, predict):
        alpha = self.alpha[-1]
        self.w *= np.exp(alpha * (label != predict))

    def fit(self, data: np.ndarray, label: np.ndarray):
        data = data.copy()
        label = label.copy()
        label[label == 0] = -1
        self.n_data = data.shape[0]

        self.w = np.array([1 / self.n_data] * self.n_data)

        for _ in range(self.T):
            clf = DecisionTreeClassifier()
            clf.fit(X=data, y=label, sample_weight=self.w)
            predicted = clf.predict(data)
            self.classifier.append(clf)

            err = self.compute_error(label, predicted)

            alpha = self.compute_alpha(err)
            self.alpha.append(alpha)
            self.update_weights(label, predicted)

        self.alpha = np.array(self.alpha)

    def predict(self, data: np.ndarray):
        predict = np.zeros(data.shape[0])
        for i in range(self.T):
            predict += self.alpha[i] * self.classifier[i].predict(data)
        predict = np.sign(predict)

        predict[predict == -1] = 0

        return predict
