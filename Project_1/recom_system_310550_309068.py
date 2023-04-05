import argparse
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
import sys
import random as rd
import math
import pandas as pd

"""
Execution:
python recom_system_310550_309068.py -tr train_x -te test_x -a SVD1 -r yes.txt
"""


def parsing():
    parser = argparse.ArgumentParser(description='Recommendation System')
    parser.add_argument('--train',
                        '-tr',
                        help='Train file with .csv format',
                        required=True)
    parser.add_argument('--test',
                        '-te',
                        help='Test file with .csv format',
                        required=True)
    parser.add_argument('--alg',
                        '-a',
                        choices=["NMF", "SVD1", "SVD2", "SGD"],
                        help='NMF or SVD1 or SVD2 or SGD',
                        required=True)
    parser.add_argument('--result',
                        '-r',
                        help='Result file in which there is final result of RMSE',
                        default='result.txt')
    return parser.parse_args()


def SVD1(matrixx, n_componentss, random_statee):
    pass


def SVD2(matrixx, n_componentss, random_statee, stopp):
    """
    while True:
        ...
        if WARUNEK_STOPU:
            break

    """
    pass


def SGD():
    pass

# output: Z' matrix, test_matrix, test_mask
def RMSE(output, test_matrix, test_mask):
    output_small = np.zeros(test_mask.shape)
    test_ile = 0
    rmse_tmp = 0
    for i in range(output_small.shape[0]):
        output_small[i,] = output[i,test_mask[i,]==1]
        test_ile = test_ile + int(np.sum(test_mask[i,]))
        rmse_tmp = rmse_tmp + np.sum(np.power(output_small[i,]-test_matrix[i,],2))
    rmse = math.sqrt(1/test_ile*rmse_tmp)
    return rmse

if __name__ == "__main__":
    args = parsing()
    trains_file = args.train
    tests_file = args.test
    alg = str(args.alg)
    result_file = args.result

    with open('./ratings.csv', 'r') as ratings, \
            open(f'{tests_file}', 'r') as test_file, \
            open(f'{trains_file}', 'r') as train_file:
        ratings = np.genfromtxt(ratings, delimiter=',')[1:] # Opening files

        matrix = np.zeros((int(np.max(ratings[:, 0])), int(np.max(ratings[:, 1]))))
        # print(matrix.shape) # = (610, 193609)
        list_for_movies = []
        # there is 9724 unique movieID's
        for row in ratings:
            matrix[int(row[0]) - 1, int(row[1]) - 1] = row[2] # Filling matrix
            if row[1] not in list_for_movies:
                list_for_movies.append(int(row[1]))
        # matrix = matrix.astype('uint8')
        list_for_movies.sort()
        dict_for_movies = {i: list_for_movies[i] for i in range(len(list_for_movies))} # May be usefull
        # print(dict_for_movies)

        pointer_train = np.zeros(matrix.shape)
        for row in train_file:
            pointer_train[int(row[0]) - 1, int(row[1]) - 1] = 1 # Creating mask for training file

        pointer_test = np.zeros(matrix.shape)
        for row in test_file:
            pointer_test[int(row[0]) - 1, int(row[1]) - 1] = 1 # Creating mask for test file

        nonzero_indices = np.nonzero(matrix)
        nonempty_columns = np.unique(nonzero_indices[1])
        empty_columns = list(set(range(matrix.shape[1])) - set(nonempty_columns)) # Searching for empty columns

        matrix_small = np.delete(matrix, empty_columns, axis=1) # Removing empty columns in main matrix
        pointer_train_small = np.delete(pointer_train, empty_columns, axis=1) # Removing empty columns in train matrix
        pointer_test_small = np.delete(pointer_test, empty_columns, axis=1) # Removing empty columns in test matrix
        print(matrix_small.shape)  # = (610, 9724)
        print(pointer_train_small.shape)
        print(pointer_test_small.shape)

    # Proceeding
    if alg == "NMF":
        result = len(alg)
    elif alg == "SVD1":
        result = len(alg)
    elif alg == "SVD2":
        result = len(alg) + 1
    elif alg == "SGD":
        result = len(alg) + 2
    else:
        sys.exit("Something went wrong.")

    # End of the script and saving a result
    with open(f'{result_file}', 'w') as res_file:
        res_file.write(f"{result}")
