import pandas as pd
import numpy as np


class DT:
    def __init__(self) -> None:
        self.data = None
        self.n_data = None
        self.label = None
        self.n_positive = None
        self.n_negative = None
        self.column = None
        self.distinct_value = None

    def read_dataset(self, file_path):
        """
        Read dataset from file_path which is a csv file.
        """

        # Read data from file_path
        dataset = pd.read_csv(file_path, sep=",")
        # Get column from data
        self.labels = dataset.columns.values
        # Get data from data and convert to numpy array
        self.data = dataset.iloc[:, :-1].to_numpy().astype(int)
        # Get number of data
        self.n_data = self.data.shape[0]
        # Find distinct value of each column
        self.distinct_value = [np.unique(self.data[:, i]) for i in range(self.data.shape[1])]
        # Get label from data and convert to numpy array
        self.label = dataset.iloc[:, -1].to_numpy().astype(bool)
        # Get number of positive and negative label
        self.n_positive = np.sum(self.label)
        self.n_negative = self.n_data - self.n_positive

        print(self.data)

    def n_data_positive_negetive(self, column_id, value):
        '''
        Return total number of data which has value in column_id 
        and number of positive and negative label.
        '''
        
        # Get label of data which has value in column_id
        label = self.label[self.data[:, column_id] == value]

        return label.shape[0], np.sum(label), label.shape[0] - np.sum(label)


    def calc_hellinger_distance(self, column_id):
        '''
        Calculate Hellinger distance of a column.
        '''

        # Get distinct value of column
        distinct_value = self.distinct_value[column_id]
        
        column_value = self.data[:, column_id]

        hellinger = 0
        for value in distinct_value:
            # Get label of data which has value in column_id
            label = self.label[column_value == value]
            n_label = label.shape[0]
            n_positive_label = np.sum(label)
            n_negative_label = label.shape[0] - n_positive_label


model = DT()
model.read_dataset("Covid19HDDT.csv")
