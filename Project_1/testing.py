import argparse
import numpy as np
from sklearn.decomposition import NMF
from sklearn.decomposition import TruncatedSVD
from sklearn.exceptions import ConvergenceWarning
import sys
import warnings
import matplotlib.pylab as plt
import matplotlib as mpl
import copy as cp

mpl.rcParams.update(mpl.rcParamsDefault)

"""
Execution:
python testing.py -tr train_x -te test_x -a SVD1
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
    return parser.parse_args()


def RMSE(matrixx, testt, test_mask):
    resultt = testt - np.multiply(np.round(2 * matrixx) / 2, test_mask)
    resultt = np.sqrt(np.sum(resultt ** 2) / np.sum(test_mask))
    return resultt


def fill_missing(matrix_data, method=0, column=0):
    """
    Method:
        - 0 - fill with zeros
        - 1 - fill with mean (in every row)
        - 2 - fill with random variable from N(mu,sd) but truncated (0,5)
        - 3 - fill with random variable, probs based on data
        - 4 - fill with the most frequent value, consider floor(r) (----row-----/column/matrix)
        - 5 
    """
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
        to_fill = mean_row*mean_column
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

    with open('./ratings.csv', 'r') as ratings, \
            open(f'{tests_file}', 'r') as test_file, \
            open(f'{trains_file}', 'r') as train_file:
        ratings = np.genfromtxt(ratings, delimiter=',')[1:]
        train_file = np.genfromtxt(train_file, delimiter=',')[1:]
        test_file = np.genfromtxt(test_file, delimiter=',')[1:]

        matrix = np.zeros((int(np.max(ratings[:, 0])), int(np.max(ratings[:, 1]))))
        for row in ratings:
            matrix[int(row[0]) - 1, int(row[1]) - 1] = row[2]

        pointer_train = np.zeros(matrix.shape)
        train = np.zeros(matrix.shape)
        for row in train_file:
            pointer_train[int(row[0]) - 1, int(row[1]) - 1] = 1
            train[int(row[0]) - 1, int(row[1]) - 1] = row[2]

        pointer_test = np.zeros(matrix.shape)
        test = np.zeros(matrix.shape)
        for row in test_file:
            pointer_test[int(row[0]) - 1, int(row[1]) - 1] = 1
            test[int(row[0]) - 1, int(row[1]) - 1] = row[2]

    nonzero_indices = np.nonzero(matrix)
    nonempty_columns = np.unique(nonzero_indices[1])
    empty_columns = list(set(range(matrix.shape[1])) - set(nonempty_columns))

    matrix_small = np.delete(matrix, empty_columns, axis=1)
    pointer_train_small = np.delete(pointer_train, empty_columns, axis=1)
    pointer_test_small = np.delete(pointer_test, empty_columns, axis=1)
    train_small = np.delete(train, empty_columns, axis=1)
    test_small = np.delete(test, empty_columns, axis=1)

    if alg == "NMF":
        rmsess = []
        matrix_temp = fill_missing(train_small, method=4)
        rmse0 = RMSE(matrix_temp, test_small, pointer_test_small)
        rmsess.append(rmse0)
        print(f"Without performing algorithm, RMSE is {rmse0}")
        warnings.filterwarnings(action='ignore', category=ConvergenceWarning)
        for i in range(1, 101):
            nmf = NMF(n_components=i, init='nndsvda', random_state=666,
                      max_iter=200)  # init = 'random', init = 'nndsvdar'
            W = nmf.fit_transform(matrix_temp)
            H = nmf.components_
            X_nmf = np.dot(W, H)
            rmse = RMSE(X_nmf, test_small, pointer_test_small)
            rmsess.append(rmse)
            print(f"For n_comp: {i}, RMSE is {rmse}")
        plt.figure(1)
        plt.scatter([x for x in range(101)], rmsess, color="blue")
        plt.title(r"RMSE for NMF, Method 4")
        plt.xlabel(r"Level of truncation $r$")
        plt.ylabel("RMSE")
        plt.xticks(np.arange(0, 101, step=1))
        plt.show()

    elif alg == "SVD1":
        rmsess = []
        matrix_temp = fill_missing(train_small, method=4)
        rmse0 = RMSE(matrix_temp, test_small, pointer_test_small)
        rmsess.append(rmse0)
        print(f"Without performing algorithm, RMSE is {rmse0}")
        for i in range(1, 31):
            SVD = TruncatedSVD(n_components=i, n_iter=1, random_state=666)
            X_svd = SVD.fit_transform(matrix_temp)
            X_svd = SVD.inverse_transform(X_svd)
            rmse = RMSE(X_svd, test_small, pointer_test_small)
            rmsess.append(rmse)
            print(f"For n_comp: {i}, RMSE is {rmse}")
        plt.figure(2)
        plt.scatter([x for x in range(31)], rmsess, color="green")
        plt.title(r"RMSE for SVD1 within Method 4")
        plt.xlabel(r"Level of truncation $r$")
        plt.ylabel("RMSE")
        plt.xticks(np.arange(0, 32, step=1))
        plt.show()

    elif alg == "SVD2":
        matrix_temp = fill_missing(matrix_small, method=0)
        result = len(alg) + 1

    elif alg == "SGD":
        r = 15  # num of components
        steps = 10  # steps to be performed
        iterations = 10000      # iterations for each step
        learning_rate = 0.05   # learning rate
        lam = 0.05      # lambda from equation
        
        it_tmp = 0 # number of current iteration
        
        non_missing = np.argwhere(train_small!=0)
        non_missing = list(map(tuple,non_missing))
        
        # initial matrices W,H
        W = np.reshape(np.random.uniform(low=0,high=0.3,size=train_small.shape[0]*r),(train_small.shape[0], r))
        H = np.reshape(np.random.uniform(1,2,size=r*train_small.shape[1]),(r, train_small.shape[1]))
    
        for iteration in range(iterations*steps):
            it_tmp = it_tmp + 1
            if it_tmp % iterations == 0:
                learning_rate = learning_rate * 0.9
                
            tmp_sample = non_missing[np.random.randint(len(non_missing))]  # sample one random point
            W_tmp = cp.deepcopy(W[tmp_sample[0], :])
            # update value
            W[tmp_sample[0],:] += learning_rate * 2 * (H[:,tmp_sample[1]] * (train_small[tmp_sample]-np.dot(W[tmp_sample[0], :], H[:,tmp_sample[1]]))-lam*W[tmp_sample[0],:])
            H[:,tmp_sample[1]] += learning_rate * 2 * (W_tmp*(train_small[tmp_sample]-np.dot(W_tmp,H[:,tmp_sample[1]]))-lam*H[:,tmp_sample[1]])
        Z_sgd = np.matmul(W,H)
        print(RMSE(Z_sgd, test_small, pointer_test_small))

    else:
        sys.exit("Something went wrong.")

