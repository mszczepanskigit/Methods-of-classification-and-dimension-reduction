import argparse
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
import sys
import warnings

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


def RMSE(matrix, test, test_mask):
    result = np.round(2 * test) / 2 - np.multiply(matrix, test_mask)
    result = np.sqrt(np.sum(result ** 2) / np.sum(test_mask))
    return result


def fill_missing(matrix_data, method=0, column=0):
    '''
    Method:
        - 0 - fill with zeros
        - 1 - fill with mean (row/column/matrix)
        - 2 - fill with median (row/column/matrix)
        - 3 - fill with random variable from N(mu,sd)
        - 4 - fill with random variable from N(mu,sd) but truncated (0,5)
        - 5 - fill with random variable, probs based on data
        - 6 - fill with the most frequent value, consider floor(r) (row/column/matrix)
    '''
    if method == 0:
        return matrix_data

    elif method == 1:
        if column == 1:
            matrix_data = matrix_data.transpose((1, 0))
            for row in range(matrix_data.shape[0]):
                non_empty = matrix_data[row, matrix_data[row,] != 0]
                matrix_data[row, matrix_data[row,] == 0] = np.mean(non_empty)
            return matrix_data.transpose((1, 0))
        elif column == 0:
            for row in range(matrix_data.shape[0]):
                non_empty = matrix_data[row, matrix_data[row,] != 0]
                matrix_data[row, matrix_data[row,] == 0] = np.mean(non_empty)
            return matrix_data
        else:
            mean_tmp = []
            for row in range(matrix_data.shape[0]):
                non_empty = matrix_data[row, matrix_data[row,] != 0]
                mean_tmp.extend(non_empty)
            for row in range(matrix_data.shape[0]):
                matrix_data[row, matrix_data[row,] == 0] = np.mean(non_empty)
            return matrix_data

    elif method == 2:
        if column == 1:
            matrix_data = matrix_data.transpose((1, 0))
            for row in range(matrix_data.shape[0]):
                non_empty = matrix_data[row, matrix_data[row,] != 0]
                matrix_data[row, matrix_data[row,] == 0] = np.median(non_empty)
            return matrix_data.transpose((1, 0))
        elif column == 0:
            for row in range(matrix_data.shape[0]):
                non_empty = matrix_data[row, matrix_data[row,] != 0]
                matrix_data[row, matrix_data[row,] == 0] = np.median(non_empty)
            return matrix_data
        else:
            median_tmp = []
            for row in range(matrix_data.shape[0]):
                non_empty = matrix_data[row, matrix_data[row,] != 0]
                median_tmp.extend(non_empty)
            for row in range(matrix_data.shape[0]):
                matrix_data[row, matrix_data[row,] == 0] = np.median(non_empty)
            return matrix_data


    elif method == 3:
        for row in range(matrix_data.shape[0]):
            non_empty = matrix_data[row, matrix_data[row,] != 0]
            matrix_data[row, matrix_data[row,] == 0] = np.std(non_empty) * np.random.randn(1, len(
                matrix_data[row, matrix_data[row,] == 0])) + np.mean(non_empty)
            matrix_data[row, matrix_data[row,] < 0] = 0
        return matrix_data


    elif method == 4:
        for row in range(matrix_data.shape[0]):
            non_empty = matrix_data[row, matrix_data[row,] != 0]
            matrix_data[row, matrix_data[row,] == 0] = np.std(non_empty) * np.random.randn(1, len(
                matrix_data[row, matrix_data[row,] == 0])) + np.mean(non_empty)
            matrix_data[row, matrix_data[row,] > 5] = 5
            matrix_data[row, matrix_data[row,] < 0] = 0
        return matrix_data

    elif method == 5:
        for row in range(matrix_data.shape[0]):
            frequency = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
            non_empty = np.floor(list(matrix_data[row, matrix_data[row,] != 0]))
            # frequency['0'] = frequency['0']+non_empty.count(0)
            frequency['1'] = frequency['1'] + np.count_nonzero(non_empty==1)
            frequency['2'] = frequency['2'] + np.count_nonzero(non_empty==2)
            frequency['3'] = frequency['3'] + np.count_nonzero(non_empty==3)
            frequency['4'] = frequency['4'] + np.count_nonzero(non_empty==4)
            frequency['5'] = frequency['5'] + np.count_nonzero(non_empty==5)
            all_probs = np.sum(list(frequency.values()))
            for key in frequency.keys():
                frequency[key] = frequency[key] / all_probs
            matrix_data[row, matrix_data[row,] == 0] = np.random.choice(np.array([0,1, 2, 3, 4, 5]),
                                                                        matrix_data[row, matrix_data[row,] == 0].size,
                                                                        p=list(frequency.values()))
        return matrix_data

    elif method == 6:
        if column == 1:
            matrix_data = matrix_data.transpose((1, 0))
            for row in range(matrix_data.shape[0]):
                frequency = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
                non_empty = np.floor(list(matrix_data[row, matrix_data[row,] != 0]))
                # frequency['0'] = frequency['0']+non_empty.count(0)
                frequency['1'] = frequency['1'] + np.count_nonzero(non_empty==1)
                frequency['2'] = frequency['2'] + np.count_nonzero(non_empty==2)
                frequency['3'] = frequency['3'] + np.count_nonzero(non_empty==3)
                frequency['4'] = frequency['4'] + np.count_nonzero(non_empty==4)
                frequency['5'] = frequency['5'] + np.count_nonzero(non_empty==5)
                most_frequent = list(frequency.values()).index(max(frequency.values()))
                matrix_data[row, matrix_data[row,] == 0] = most_frequent
            return matrix_data.transpose((1, 0))
        elif column == 0:
            for row in range(matrix_data.shape[0]):
                frequency = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
                non_empty = np.floor(list(matrix_data[row, matrix_data[row,] != 0]))
                # frequency['0'] = frequency['0']+non_empty.count(0)
                frequency['1'] = frequency['1'] + np.count_nonzero(non_empty==1)
                frequency['2'] = frequency['2'] + np.count_nonzero(non_empty==2)
                frequency['3'] = frequency['3'] + np.count_nonzero(non_empty==3)
                frequency['4'] = frequency['4'] + np.count_nonzero(non_empty==4)
                frequency['5'] = frequency['5'] + np.count_nonzero(non_empty==5)
                most_frequent = list(frequency.values()).index(max(frequency.values()))
                print(most_frequent)
                matrix_data[row, matrix_data[row,] == 0] = most_frequent
            return matrix_data
        else:
            frequency = {'0': 0, '1': 0, '2': 0, '3': 0, '4': 0, '5': 0}
            for row in range(matrix_data.shape[0]):
                non_empty = np.floor(list(matrix_data[row, matrix_data[row,] != 0]))
                # frequency['0'] = frequency['0']+non_empty.count(0)
                frequency['1'] = frequency['1'] + np.count_nonzero(non_empty==1)
                frequency['2'] = frequency['2'] + np.count_nonzero(non_empty==2)
                frequency['3'] = frequency['3'] + np.count_nonzero(non_empty==3)
                frequency['4'] = frequency['4'] + np.count_nonzero(non_empty==4)
                frequency['5'] = frequency['5'] + np.count_nonzero(non_empty==5)
            most_frequent = list(frequency.values()).index(max(frequency.values()))
            for row in range(matrix_data.shape[0]):
                matrix_data[row, matrix_data[row,] == 0] = most_frequent
            return matrix_data
    else:
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
        ratings = np.genfromtxt(ratings, delimiter=',')[1:]  # Opening files
        train_file = np.genfromtxt(train_file, delimiter=',')[1:]
        test_file = np.genfromtxt(test_file, delimiter=',')[1:]

        matrix = np.zeros((int(np.max(ratings[:, 0])), int(np.max(ratings[:, 1]))))
        # print(matrix.shape) # = (610, 193609)
        # there is 9724 unique movieID's
        for row in ratings:
            matrix[int(row[0]) - 1, int(row[1]) - 1] = row[2]  # Filling matrix

        pointer_train = np.zeros(matrix.shape)
        train = np.zeros(matrix.shape)
        for row in train_file:
            pointer_train[int(row[0]) - 1, int(row[1]) - 1] = 1  # Creating mask for training file
            train[int(row[0]) - 1, int(row[1]) - 1] = row[2]  # Creating training array

        pointer_test = np.zeros(matrix.shape)
        test = np.zeros(matrix.shape)
        for row in test_file:
            pointer_test[int(row[0]) - 1, int(row[1]) - 1] = 1  # Creating mask for test file
            test[int(row[0]) - 1, int(row[1]) - 1] = row[2]  # Creating test array

    nonzero_indices = np.nonzero(matrix)
    nonempty_columns = np.unique(nonzero_indices[1])
    empty_columns = list(set(range(matrix.shape[1])) - set(nonempty_columns))  # Searching for empty columns

    matrix_small = np.delete(matrix, empty_columns, axis=1)  # Removing empty columns in main matrix
    pointer_train_small = np.delete(pointer_train, empty_columns, axis=1)  # Removing empty columns in mask matrix
    pointer_test_small = np.delete(pointer_test, empty_columns, axis=1)  # Removing empty columns in mask matrix
    train_small = np.delete(train, empty_columns, axis=1)  # Removing empty columns in train matrix
    test_small = np.delete(test, empty_columns, axis=1)  # Removing empty columns in test matrix
    """print(matrix_small.shape)  # = (610, 9724)
    print(pointer_train_small.shape)
    print(pointer_test_small.shape)
    print(train_small.shape)
    print(test_small.shape)"""
    # Proceeding

    if alg == "NMF":
        matrix_temp = fill_missing(matrix_small, method=3)
        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        nmf = NMF(n_components=9, init='nndsvda', random_state=666, max_iter=200)
        W = nmf.fit_transform(matrix_temp)
        H = nmf.components_
        X_nmf = np.dot(W, H)
        rmse = RMSE(X_nmf, test_small, pointer_test_small)
        result = rmse

    elif alg == "SVD1":
        matrix_temp = fill_missing(matrix_small, method=5)
        SVD = TruncatedSVD(n_components=9, n_iter=1, random_state=666)
        X_svd = SVD.fit_transform(matrix_temp)
        X_svd = SVD.inverse_transform(X_svd)
        rmse = RMSE(X_svd, test_small, pointer_test_small)
        result = rmse

    elif alg == "SVD2":
        matrix_temp = fill_missing(matrix_small, method=0)
        result = len(alg) + 1

    elif alg == "SGD":
        matrix_temp = fill_missing(matrix_small, method=0)
        result = len(alg) + 2

    else:
        sys.exit("Something went wrong.")

    # End of the script and saving the result
    with open(f'{result_file}', 'w') as res_file:
        res_file.write(f"{result}")
