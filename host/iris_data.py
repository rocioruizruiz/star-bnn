# Author: Laurentiu-Cristian Duca
# license: BSD3

import pandas as pd
import numpy as np

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Setosa', 'Versicolor', 'Virginica']

def load_data():

    train_path = "iris_training_3_outputs.csv"
    test_path = "iris_test_3_outputs.csv"

    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)

    # convert from DataFrame to numpy.ndarray
    train = train.values
    train_x = train[:,0:4]
    train_y = train[:,4:]

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    # convert from DataFrame to numpy.ndarray
    test = test.values
    test_x = test[:,0:4]
    test_y = test[:,4:]
    #print(test_x)
    #print(test_y)

    return (train_x, train_y), (test_x, test_y)

