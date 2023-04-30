import argparse
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
import sys
import warnings
import copy as cp

"""
Execution:
python recom_system_310550_309068.py -tr train_x -te test_x -a SVD1 -r result.txt

python recom_system_310550_309068.py --train train_x --test test_x --alg SVD1 --result result.txt
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


def fill_missing(matrix_data_input, method=0, column=0):
    """
    Method:
        - 0 - fill with zeros
        - 1 - fill with mean (in every row)
        - 2 - fill with random variable from N(mu,sd) but truncated to (0,5)
        - 3 - fill with random variable, probs based on data
        - 4 - fill with the most frequent value, consider floor(r) (in every row)
        - 5 - fill with weighted average
    """
    matrix_data = cp.deepcopy(matrix_data_input)
    
    if method == 0:
        return matrix_data

    elif method == 1:
        if column == 1:
            matrix_data = matrix_data.transpose((1, 0))
            for row in range(matrix_data.shape[0]):
                non_empty = matrix_data[row, matrix_data[row, ] != 0]
                matrix_data[row, matrix_data[row, ] == 0] = np.mean(non_empty)
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
        for row in range(matrix_data.shape[0]):
            non_empty = matrix_data[row, matrix_data[row,] != 0]
            matrix_data[row, matrix_data[row,] == 0] = np.std(non_empty) * np.random.randn(1, len(
                matrix_data[row, matrix_data[row,] == 0])) + np.mean(non_empty)
            matrix_data[row, matrix_data[row,] > 5] = 5
            matrix_data[row, matrix_data[row,] < 0] = 0
        return matrix_data

    elif method == 3:
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

    elif method == 4:
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
        
    elif method == 5:
        mean_row = np.zeros(matrix_data.shape)
        mean_column = np.zeros(matrix_data.shape)
        missing = np.argwhere(matrix_data==0)
        missing = list(map(tuple,missing))
        for row in range(matrix_data.shape[0]):
            non_empty = np.floor(list(matrix_data[row, matrix_data[row,] != 0]))
            mean_row[row,:] = np.mean(non_empty)
        matrix_data = matrix_data.transpose((1, 0))
        mean_column = mean_column.transpose((1,0))
        for row in range(matrix_data.shape[0]):
            non_empty = np.floor(list(matrix_data[row, matrix_data[row,] != 0]))
            if(len(non_empty)==0):
                mean_column[row,:] = 0
            else:
                mean_column[row,:] = np.mean(non_empty)
        mean_column = mean_column.transpose((1,0))
        to_fill = 0.66*mean_row + 0.34*mean_column
        matrix_data = matrix_data.transpose((1, 0))
        for ind in missing:
            i,j = ind
            matrix_data[i,j] = to_fill[i,j]
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
        ratings = np.genfromtxt(ratings, delimiter=',')[1:]  # Opening files without header
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

    # Proceeding
    if alg == "NMF":
        matrix_temp = fill_missing(train_small, method=5)
        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        nmf = NMF(n_components=27, init='nndsvda', random_state=666, max_iter=200)
        W = nmf.fit_transform(matrix_temp)
        H = nmf.components_
        X_nmf = np.dot(W, H)
        rmse = RMSE(X_nmf, test_small, pointer_test_small)
        result = rmse

    elif alg == "SVD1":
        matrix_temp = fill_missing(train_small, method=5)
        SVD = TruncatedSVD(n_components=5, n_iter=1, random_state=666)
        X_svd = SVD.fit_transform(matrix_temp)
        X_svd = SVD.inverse_transform(X_svd)
        rmse = RMSE(X_svd, test_small, pointer_test_small)
        result = rmse

    elif alg == "SVD2":
        matrix_temp = fill_missing(train_small, method=5)
        r = 5      # number of components
        steps_max = 9  # maximum number of iterations to be performed
        step_it = 0     # current step
        rmsess = []
        no_train = np.zeros(pointer_train_small.shape)+1-pointer_train_small
        while step_it < steps_max:
            previous = cp.deepcopy(matrix_temp)
            step_it = step_it + 1
            SVD = TruncatedSVD(n_components=r, n_iter=1, random_state=666)
            X_svd = SVD.fit_transform(matrix_temp)
            X_svd = SVD.inverse_transform(X_svd)   # result of SVD
            X_new = cp.deepcopy(train_small)
            # replace elements outside the training set with obtained values
            X_new = X_new + no_train*X_svd
            matrix_temp = cp.deepcopy(X_new)
            rmsess.append(RMSE(X_new, test_small, pointer_test_small))
        result = rmsess[-1]
       

    elif alg == "SGD":
        np.random.seed(666)
        r = 1  # num of components
        steps = 35  # steps to be performed
        iterations = 50000      # iterations for each step
        learning_rate = 0.007   # learning rate
        lam = 0.00001   # lambda from equation
        factor = 0.88
        it_tmp = 0 # number of current iteration
        non_missing = np.argwhere(train_small!=0)
        non_missing = list(map(tuple,non_missing))
        # initial matrices W,H
        W = np.reshape(np.random.uniform(low=0,high=0.2,size=train_small.shape[0]*r),(train_small.shape[0], r))
        H = np.reshape(np.random.uniform(0.9,2.8,size=r*train_small.shape[1]),(r, train_small.shape[1]))
        for iteration in range(iterations*steps):
            it_tmp = it_tmp + 1
            if it_tmp % iterations == 0:
                learning_rate = learning_rate * factor
                    
            tmp_sample = non_missing[np.random.randint(len(non_missing))]  # sample one random point
            W_tmp = cp.deepcopy(W[tmp_sample[0], :])
            # update value
            W[tmp_sample[0],:] += learning_rate * 2 * (H[:,tmp_sample[1]] * (train_small[tmp_sample]-np.dot(W[tmp_sample[0], :], H[:,tmp_sample[1]]))-lam*W[tmp_sample[0],:])
            H[:,tmp_sample[1]] += learning_rate * 2 * (W_tmp*(train_small[tmp_sample]-np.dot(W_tmp,H[:,tmp_sample[1]]))-lam*H[:,tmp_sample[1]])
        Z_sgd = np.matmul(W,H)
        rmse = RMSE(Z_sgd, test_small, pointer_test_small)
        result = rmse
        

    else:
        sys.exit("Something went wrong.")

    # End of the script and saving the result
    with open(f'{result_file}', 'w') as res_file:
        res_file.write(f"{result}")
