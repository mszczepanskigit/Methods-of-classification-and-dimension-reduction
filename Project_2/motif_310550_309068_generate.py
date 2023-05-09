import json
import numpy as np
import argparse


def ParseArguments():
    parser = argparse.ArgumentParser(description="Motif generator")
    parser.add_argument('--params',
                        '-p',
                        default="params_set_3x10.json",
                        required=False,
                        help='Parameters file (default: %(default)s)')
    parser.add_argument('--output',
                        '-o',
                        default="data_3x10.json",
                        required=False,
                        help='Generated data with specified parameters (default: %(default)s)')
    args = parser.parse_args()
    return args.params, args.output


param_file, output_file = ParseArguments()
#param_file = 'params_set_3x10.json'

with open(param_file, 'r') as inputfile:
    params = json.load(inputfile)

w = params['w']
k = params['k']
alpha = params['alpha']
Theta = np.asarray(params['Theta'])
ThetaB = np.asarray(params['ThetaB'])

# TO DO: wywymulowac x_1, ..., x_k, gdzie x_i=(x_{i1},..., x_{iw})
# zgodnie z opisem jak w skrypcie i zapisac w pliku .csv
# i-ta linijka = x_i

# Dla przykladu, zalozmy, ze x_1,... x_k sa zebrane w macierz X
# (dla przypomnienia: kazdy x_{ij} to A,C,G lub T, co utozsamiamy z 1,2,3,4

X = []
theta_prim = Theta.transpose()

for i in range(k):
    z_tmp = np.random.choice([0, 1], 1, p=[1 - alpha, alpha])
    if z_tmp == 1:  # motif
        x_tmp = []
        for j in range(w):
            x_tmp.append(np.random.choice([1, 2, 3, 4], 1, p=theta_prim[j,])[0])
        X.append(x_tmp)
    elif z_tmp == 0:  # not_motif
        X.append(list(np.random.choice([1, 2, 3, 4], w, p=ThetaB)))
    else:
        pass

# Musimy zapisac powyzszy X oraz alpha (k i w mozna potem odczytac z X)

gen_data = {
    "alpha": alpha,
    "X": [[int(x) for x in sublist] for sublist in X]
}
#output_file = 'w.json'

with open(output_file, 'w') as outfile:
    json.dump(gen_data, outfile)
