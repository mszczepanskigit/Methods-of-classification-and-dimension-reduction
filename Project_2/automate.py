import numpy as np
import sys
import math
import pandas as pd
import csv

# w = [3, 4, 5, 6, 7, 8, 9]
# k = [10, 11, 12, 13]

name = 'alpha1_test_2-0.9'

w = [10]
k = 50*[1000]

estimate_alpha = 'yes'
methodd = 0

iters = 100

alpha_list = [0.9]
np.random.seed(666)
#alf = np.random.uniform(0, 1)
alf = 0.5


def generate_theta(w):
    lst = []
    for _ in range(w):
        a, c, g, t = np.random.random(4)
        s = a + c + g + t
        a1, c1, g1, t1 = a / s, c / s, g / s, t / s
        lst.append([a1, c1, g1, t1])

    tmp = np.array(lst)
    Theta = tmp.T

    a, c, g, t = np.random.random(4)
    s = a + c + g + t
    a1, c1, g1, t1 = a / s, c / s, g / s, t / s
    Theta_b = [a1, c1, g1, t1]

    return [Theta_b, Theta]


def generate_data(Theta, Theta_b, k, alpha):
    X = []
    theta_prim = Theta.transpose()

    for i in range(k):
        z_tmp = np.random.choice([0, 1], 1, p=[1 - alpha, alpha])
        if z_tmp == 1:  # motif
            x_tmp = []
            for j in range(Theta.shape[1]):
                x_tmp.append(np.random.choice([1, 2, 3, 4], 1, p=theta_prim[j,])[0])
            X.append(x_tmp)
        elif z_tmp == 0:  # not_motif
            X.append(list(np.random.choice([1, 2, 3, 4], Theta.shape[1], p=Theta_b)))
        else:
            pass

    return np.array(X)


def ThetaB_func(X, alpha, method=methodd):
    k, w = X.shape
    if method == 0:
        A = (X == 1).sum()
        C = (X == 2).sum()
        G = (X == 3).sum()
        T = (X == 4).sum()
        assert np.isclose(math.floor(k * w), math.floor(A + C + G + T))
        ThetaB = np.asarray([A, C, G, T]) / (k * w)
        return ThetaB
    elif method == 1:
        quant_of_rows = np.max([math.floor((1 - alpha) * k), 1])
        taken_rows = np.random.choice(k, quant_of_rows, replace=False)
        X_s = X[taken_rows]
        assert X_s.shape[0] == quant_of_rows
        A = (X_s == 1).sum()
        C = (X_s == 2).sum()
        G = (X_s == 3).sum()
        T = (X_s == 4).sum()
        assert np.isclose(A + C + G + T, math.floor(quant_of_rows * w))
        ThetaB = np.asarray([A, C, G, T]) / (quant_of_rows * w)
        return ThetaB
    elif method == 2:
        ThetaB = np.array([0.25, 0.25, 0.25, 0.25])
        return ThetaB
    else:
        sys.exit("Error with filling ThetaB method")


def Theta_func(X, alpha, method=methodd):
    k, w = X.shape
    Theta = np.zeros((4, w))
    if method == 0:
        for position in range(w):
            column = X[:, position]
            A = (column == 1).sum()
            C = (column == 2).sum()
            G = (column == 3).sum()
            T = (column == 4).sum()
            assert np.isclose(A + C + G + T, k)
            Theta[0, position] = A / k
            Theta[1, position] = C / k
            Theta[2, position] = G / k
            Theta[3, position] = T / k
        return Theta
    elif method == 1:
        quant_of_rows = np.max([math.floor(alpha * k), 1])
        taken_rows = np.random.choice(k, quant_of_rows, replace=False)
        X_s = X[taken_rows]
        for position in range(w):
            column = X_s[:, position]
            A = (column == 1).sum()
            C = (column == 2).sum()
            G = (column == 3).sum()
            T = (column == 4).sum()
            assert np.isclose((A + C + G + T), quant_of_rows)
            Theta[0, position] = A / quant_of_rows
            Theta[1, position] = C / quant_of_rows
            Theta[2, position] = G / quant_of_rows
            Theta[3, position] = T / quant_of_rows
        return Theta
    elif method == 2:
        return Theta + 0.25
    else:
        sys.exit("Error with filling Theta method")


def Qi_func(X, alpha, Theta, ThetaB, ask=0):
    if ask == 0:
        k, w = X.shape
        Qi = np.zeros((2, k))
        for i in np.arange(k):
            tmp = np.ones(w)
            tmpB = np.ones(w)
            for j in np.arange(w):
                tmp[j] = Theta[X[i, j] - 1, j]
                tmpB[j] = ThetaB[X[i, j] - 1]

            Qi[0, i] = (1 - alpha) * tmpB.prod()
            Qi[1, i] = alpha * tmp.prod()
            Qi[:, i] = Qi[:, i] / (Qi[0, i] + Qi[1, i])
        return Qi
    else:
        k, w = X.shape
        Qi = np.zeros((2, k))
        for i in np.arange(k):
            tmp = np.ones(w)
            tmpB = np.ones(w)
            for j in np.arange(w):
                tmp[j] = Theta[int(X[i, j] - 1), j]
                tmpB[j] = ThetaB[int(X[i, j] - 1)]
            Qi[0, i] = (1 - alpha) * tmpB.prod()
            # print(f"Q_i: {Qi[0, i]}, 1-alfa: {1-alpha}, tmpB_prod: {tmpB.prod()}")
            Qi[1, i] = alpha * tmp.prod()
            Qi[:, i] = Qi[:, i] / (Qi[0, i] + Qi[1, i])
        return Qi


def dtv(p, q):
    return (np.abs(p - q)).sum() / 2


def final_dtv(Theta_org, ThetaB_org, Theta_est, ThetaB_est):
    w = len(Theta_org[1, :])
    result = dtv(ThetaB_org, ThetaB_est)
    for j in np.arange(w):
        result += dtv(Theta_org[:, j], Theta_est[:, j])
    return result / (1 + w)


def EM(X, alpha, steps=iters, m=methodd):
    dTV_tmp = []
    k, w = X.shape
    Theta = Theta_func(X, alpha, method=m)
    ThetaB = ThetaB_func(X, alpha, method=m)
    dtv_Theta_previous = 10
    p = 0
    while p < steps:
        Qi = Qi_func(X, alpha, Theta, ThetaB)
        lambB = w * Qi[0, :].sum()
        lamb = Qi[1, :].sum()
        New_ThetaB = np.ones(4)
        for letter in range(4):
            tmp = 0
            for i in np.arange(k):
                tmp = tmp + Qi[0, i] * (X[i, :] == letter + 1).sum()
            New_ThetaB[letter] = tmp / lambB

        New_Theta = np.ones((4, w))
        for letter in range(4):
            for j in range(w):
                tmp = 0
                for i in np.arange(k):
                    tmp = tmp + Qi[1, i] * (X[i, j] == letter + 1)
                New_Theta[letter, j] = tmp / lamb

        dtv_Theta_next = final_dtv(Theta, ThetaB, New_Theta, New_ThetaB)
        """if dtv_Theta_previous - dtv_Theta_next < 1 / 1000:
            break"""

        Theta = New_Theta
        ThetaB = New_ThetaB
        dTV_tmp.append(dtv_Theta_next)
        dtv_Theta_previous = dtv_Theta_next
        p += 1
    return p, Theta, ThetaB, dTV_tmp, alpha


def EM_with_alpha(X, steps=iters, m=methodd):
    dTV_tmp = []
    k, w = X.shape
    alpha = alf
    Theta = Theta_func(X, alpha, method=m)
    ThetaB = ThetaB_func(X, alpha, method=m)
    dtv_Theta_previous = 10
    dtv_alpha_previous = 10
    p = 0
    while p < steps:
        Qi = Qi_func(X, alpha, Theta, ThetaB)
        lambB = w * Qi[0, :].sum()
        lamb = Qi[1, :].sum()
        New_ThetaB = np.ones(4)
        for letter in range(4):
            tmp = 0
            for i in np.arange(k):
                tmp = tmp + Qi[0, i] * (X[i, :] == letter + 1).sum()
            New_ThetaB[letter] = tmp / lambB

        New_Theta = np.ones((4, w))
        for letter in range(4):
            for j in range(w):
                tmp = 0
                for i in np.arange(k):
                    tmp = tmp + Qi[1, i] * (X[i, j] == letter + 1)
                New_Theta[letter, j] = tmp / lamb

        New_alpha = Qi[1, :].sum() / k

        dtv_Theta_next = final_dtv(Theta, ThetaB, New_Theta, New_ThetaB)
        dtv_alpha_next = dtv(alpha, New_alpha)
        """if (dtv_Theta_previous - dtv_Theta_next < 1 / 1000 and
                dtv_alpha_previous - dtv_alpha_next < 1 / 100):
            break"""

        Theta = New_Theta
        ThetaB = New_ThetaB
        dTV_tmp.append(dtv_Theta_next)
        dtv_Theta_previous = dtv_Theta_next
        dtv_alpha_previous = dtv_alpha_next
        alpha = New_alpha
        p += 1
    return p, Theta, ThetaB, dTV_tmp, alpha


if __name__ == "__main__":
    dataframe = pd.DataFrame(np.zeros((len(w) * len(k) * len(alpha_list), 6)))
    dataframe.columns = ['w', 'k', 'init_alpha', 'alpha', 'iters', 'final_dtv']
    dataframe['w'] = dataframe['w'].astype(int)
    dataframe['k'] = dataframe['k'].astype(int)
    dataframe['init_alpha'] = dataframe['init_alpha'].astype(float)
    dataframe['alpha'] = dataframe['alpha'].astype(float)
    dataframe['iters'] = dataframe['iters'].astype(int)
    dataframe['final_dtv'] = dataframe['final_dtv'].astype(float)
    # dataframe_param3 = []
    dataframe_param3 = pd.DataFrame(np.zeros((len(w) * len(k) * len(alpha_list), 1)))
    i = 0
    for W in w:
        for K in k:
            for a in alpha_list:
                print(f"Iteration {i} out of {(len(w) * len(k) * len(alpha_list))}")
                T = generate_theta(W)
                X = generate_data(T[1], T[0], K, a)
                if estimate_alpha == 'no':
                    params = EM(X, a)
                    fdt = final_dtv(T[1], T[0], params[1], params[2])
                    dataframe.iloc[i] = [W, K, a, estimate_alpha, params[0], fdt]
                    dataframe_param3.iloc[i] = params[3]
                else:
                    params = EM_with_alpha(X)
                    fdt = final_dtv(T[1], T[0], params[1], params[2])
                    dataframe.iloc[i] = [W, K, a, params[4], params[0], fdt]
                    dataframe_param3.iloc[i] = np.abs(params[4]-a)
                i += 1

    #dataframe.to_csv(f"{name}_dataframe.csv")
    dataframe_param3.to_csv(f"{name}_alpha_results.csv")
