import argparse
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
import sys
import random as rd
import math

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
    rd.seed(0)
    args = parsing()
    trains_file = args.train
    tests_file = args.test
    alg = str(args.alg)
    result_file = args.result

    with open('./ratings.csv', 'r') as ratings, \
            open(f'{tests_file}', 'w') as test_file, \
            open(f'{trains_file}', 'w') as train_file:
        ratings = np.genfromtxt(ratings, delimiter=',')[1:]

        matrix = np.zeros((int(np.max(ratings[:, 0])), int(np.max(ratings[:, 1]))))
        # print(matrix.shape) # = (610, 193609)
        for row in ratings:
            matrix[int(row[0]) - 1, int(row[1]) - 1] = row[2]
        # matrix = matrix.astype('uint8')
        print(matrix)  # it's filled
        """test_file.write("userId,movieId,rating,timestamp\n")
        train_file.write("userId,movieId,rating,timestamp\n")
        i = 1
        for user in matrix:
            movie_temp = [movieID for movieID in range(len(user)) if matrix[i - 1, movieID] > 0]
            train = rd.sample(movie_temp, 1 + math.floor(0.9 * len(movie_temp)))
            train.sort()
            test = list(set(movie_temp).difference(set(train)))
            test.sort()
            for train_movie in train:
                train_file.write(f"{i},{train_movie},{matrix[i - 1, train_movie]},1\n")
            for test_movie in test:
                test_file.write(f"{i},{test_movie},{matrix[i - 1, test_movie]},1\n")
            i += 1"""

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
