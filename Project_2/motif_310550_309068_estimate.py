import json
import numpy as np
import argparse


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--input',
                        '-i',
                        default="generated_data.json",
                        required=False,
                        help='data_file  (default: %(default)s)')
    parser.add_argument('--output',
                        '-o',
                        default="estimated_params.json",
                        required=False,
                        help='Output file with parameters  (default: %(default)s)')
    parser.add_argument('--estimate-alpha',
                        '-a',
                        default="no",
                        required=False,
                        help='Estimate alpha, or not  (default: %(default)s)')
    args = parser.parse_args()
    return args.input, args.output, args.estimate_alpha


input_file, output_file, estimate_alpha = ParseArguments()

# input_file = 'data_3x10.json'
# output_file = 'result_3x10.json'
# estimate_alpha = 'no'


with open(input_file, 'r') as inputfile:
    data = json.load(inputfile)

alpha = data['alpha']
X = np.asarray(data['X'])
k, w = X.shape

# Theta0 = vector of size 'w'
# Theta = matrix of size 'd' x 'w'


# methods for initial thetaB
'''
    Method:
        - 0 - fill with probability of obtaining particular value based on all given data
        - 1 - fill with probability of obtaining particular value based on randomly choosed  sequences
                in case alpha - known: we choose approximately (1-alpha)*w columns
                can be done also for unknown alpha (but need to estimate alpha first)
        - 2 - uniform distribution
'''


def ThetaB_func(X, method=0):
    if method == 0:
        return np.asarray([(X == i + 1).sum() / (k * w) for i in range(4)])
    elif method == 1:
        pass
    elif method == 2:
        pass
    else:
        pass


# methods for initial thetaA
def Theta_func(X, method=0):
    Theta = np.zeros((4, w))
    for j in range(w):
        Theta[:, j] = np.asarray([(X[:, j] == i + 1).sum() / k for i in range(4)])
    return Theta


def Qi_func(X, alpha, Theta, ThetaB):
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


def dtv(p, q):
    return (np.abs(p - q)).sum() / 2


def final_dtv(Theta_org, ThetaB_org, Theta_est, ThetaB_est):
    w = len(Theta_org[1, :])
    result = dtv(ThetaB_org, ThetaB_est)
    for j in np.arange(w):
        result += dtv(Theta_org[:, j], Theta_est[:, j])
    return result / (1 + w)


def EM(X, alpha, steps=500):
    k, w = X.shape
    Theta = Theta_func(X)
    ThetaB = ThetaB_func(X)
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
        if dtv_Theta_previous - dtv_Theta_next < 1 / 1000:
            break

        Theta = New_Theta
        ThetaB = New_ThetaB
        dtv_Theta_previous = dtv_Theta_next
        p += 1
    return p, Theta, ThetaB


def EM_with_alpha(X, steps=500):
    k, w = X.shape
    Theta = Theta_func(X)
    ThetaB = ThetaB_func(X)
    alpha = np.random.uniform(0, 1)
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
        if (dtv_Theta_previous - dtv_Theta_next < 1 / 1000 and
                dtv_alpha_previous - dtv_alpha_next < 1 / 1000): break

        Theta = New_Theta
        ThetaB = New_ThetaB
        dtv_Theta_previous = dtv_Theta_next
        dtv_alpha_previous = dtv_alpha_next
        alpha = New_alpha
        p += 1
    return (p, Theta, ThetaB, alpha)


if __name__ == "__main__":
    if estimate_alpha == "yes":
        p, Theta, ThetaB, alpha = EM_with_alpha(X)
    else:
        p, Theta, ThetaB = EM(X, alpha)

    estimated_params = {
        "alpha": alpha,  # "przepisujemy" to alpha, one nie bylo estymowane
        "Theta": Theta.tolist(),  # westymowane
        "ThetaB": ThetaB.tolist()  # westymowane
    }

    with open(output_file, 'w') as outfile:
        json.dump(estimated_params, outfile)


