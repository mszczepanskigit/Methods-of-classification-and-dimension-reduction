import argparse
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
import sys


"""
Questions:
1. Strona 3 i result_file
2. Wywoływanie plików z command line, czy będą one w obecnym directory?
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


if __name__ == "__main__":
    args = parsing()
    trains_file = args.train
    tests_file = args.test
    alg = str(args.alg)
    result_file = args.result

    with open('./ratings.csv', 'r') as ratings, \
            open(f'{tests_file}', 'r') as test_file, \
            open(f'{trains_file}', 'r') as train_file:
        ratings = np.genfromtxt(ratings, delimiter=',')[1:]
        test_file = np.genfromtxt(test_file, delimiter=',')[1:]
        train_file = np.genfromtxt(train_file, delimiter=',')[1:]

        matrix = np.zeros((int(np.max(ratings[:, 0])), int(np.max(ratings[:, 1]))))
        # print(matrix.shape) # = (610, 193609)
        for row in ratings:
            matrix[int(row[0]) - 1, int(row[1]) - 1] = row[2]
        print(matrix) # it's filled

        # Proceeding
        if alg == "NMF":
            result = len(alg)
        elif alg == "SVD1":
            result = len(alg)
        elif alg == "SVD2":
            result = len(alg)+1
        elif alg == "SGD":
            result = len(alg)+2
        else:
            sys.exit("Something went wrong.")

        # End of the script and saving a result
        with open(f'{result_file}', 'w') as res_file:
            res_file.write(f"{result}")
